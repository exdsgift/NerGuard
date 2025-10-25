import json
import ollama
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pydantic import BaseModel, Field
from typing import List, Optional

# * Pretrained model encoder upload

dabertav3 = "models/dabertav3_20251019_171835"  # model folder path
tokenizer = AutoTokenizer.from_pretrained(dabertav3)
deberta_model = AutoModelForTokenClassification.from_pretrained(dabertav3)
ner_pipeline = pipeline("ner", model=deberta_model, tokenizer=tokenizer, aggregation_strategy="simple")

# * JSON Structure verification with Pydantic

class PIIVerification(BaseModel):
   """Schema Pydantic per output strutturato Llama"""

   is_valid_pii: bool = Field(description="The detected entity is a PII")
   corrected_type: Optional[str] = Field(
      description="Tipo corretto se diverso da quello predetto", default=None
   )
   confidence_score: float = Field(
      description="Test confidence int. (0-1)", ge=0.0, le=1.0
   )
   reasoning: str = Field(description="Explenation, Lang: EN")


# todo: Config file in a different file

CONFIDENCE_THRESHOLD = 0.30  # * Modify this!
OLLAMA_MODEL = "llama3.2:3b"

PII_TYPES = {
    "NAME": "Full name of persons",
    "FIRST_NAME": "First name",
    "LAST_NAME": "Last name",
    "EMAIL": "Email addresses",
    "PHONE_NUMBER": "Phone numbers",
    "STREET_ADDRESS": "Physical addresses",
    "SSN": "Social Security Number",
    "IBAN": "International Bank Account Number",
    "BBAN": "Basic Bank Account Number",
    "CREDIT_CARD_NUMBER": "Credit card numbers",
    "CREDIT_CARD_SECURITY_CODE": "Credit card security code (CVV)",
    "DRIVER_LICENSE_NUMBER": "Driver's license numbers",
    "PASSPORT_NUMBER": "Passport numbers",
    "DATE_OF_BIRTH": "Dates of birth",
    "DATE": "Generic dates",
    "DATE_TIME": "Dates and times",
    "TIME": "Times",
    "COMPANY": "Company/organization names",
    "CUSTOMER_ID": "Customer IDs",
    "EMPLOYEE_ID": "Employee IDs",
    "USER_NAME": "Usernames",
    "PASSWORD": "Passwords",
    "API_KEY": "API keys",
    "ACCOUNT_PIN": "Account PINs",
    "BANK_ROUTING_NUMBER": "Bank routing numbers",
    "SWIFT_BIC_CODE": "SWIFT/BIC codes",
    "IPV4": "IPv4 addresses",
    "IPV6": "IPv6 addresses",
    "LOCAL_LATLNG": "Geographic coordinates (latitude/longitude)",
}

# * Extraction and evaluation


def extract_entities_with_deberta(text: str):
   """
   Extracting PII ent with DeBERTaV3 and classify for confidence level.
   """
   ner_results = ner_pipeline(text)

   confident_entities = []
   uncertain_entities = []

   for entity in ner_results:
      entity_info = {
         "text": entity["word"],
         "type": entity["entity_group"],
         "confidence": float(entity["score"]),
         "start": entity["start"],
         "end": entity["end"],
         "context_window": text[
               max(0, entity["start"] - 50) : min(len(text), entity["end"] + 50)
         ],
      }

      if entity["score"] < CONFIDENCE_THRESHOLD:
         uncertain_entities.append(entity_info)
      else:
         confident_entities.append(entity_info)

   return confident_entities, uncertain_entities


def create_verification_prompt(text: str, entity: dict) -> str:
   """
   Prompt context for LLM verification
   """
   pii_descriptions = "\n".join([f"- {k}: {v}" for k, v in PII_TYPES.items()])

   prompt = f"""
   You are an expert in privacy and personal data protection. Your task is to verify whether an automatically identified entity is actually Personally Identifiable Information (PII). You don't have to miss even if a single PII entity.

   ORIGINAL CONTEXT:
   "{entity["context_window"]}"

   ENTITY TO VERIFY:
   - Text: "{entity["text"]}"
   - Predicted type: {entity["type"]}
   - Model confidence: {entity["confidence"]:.2%}

   VALID PII TYPES:
   {pii_descriptions}

   EVALUATE:
   1. Is the entity "{entity["text"]}" actually PII in the provided context?
   2. Is the type "{entity["type"]}" correct? If not, which type is more appropriate?
   3. Consider the surrounding context to avoid false positives.
   4. If the proper type is not in the provided list, choose the closest type in the list!

   Respond in JSON format following exactly this schema.
   """

   return prompt


def verify_with_llama_local(entity: dict, text: str) -> dict:
   """
   Verifica entità usando Llama 3B locale tramite Ollama con structured output
   """
   prompt = create_verification_prompt(text, entity)

   try:
      # Ollama with structured output with Pydantic schema
      response = ollama.chat(
         model=OLLAMA_MODEL,
         messages=[
               {
                  "role": "system",
                  "content": "You are an expert in privacy and personal data protection. Answare only in valid JSON.",
               },
               {"role": "user", "content": prompt},
         ],
         format=PIIVerification.model_json_schema(),  # Schema enforcement
         options={
               "temperature": 0.1,  # Bassa per output deterministico
               "num_predict": 300,  # Limite token output
         },
      )

      # Parse JSON response
      verification_result = json.loads(response["message"]["content"])

      # Validazione con Pydantic
      validated_result = PIIVerification(**verification_result)

      return validated_result.model_dump()

   except json.JSONDecodeError as e:
      print(f"Errore parsing JSON da Llama: {e}")
      # Fallback: considera valida l'entità originale
      return {
         "is_valid_pii": True,
         "corrected_type": None,
         "confidence_score": entity["confidence"],
         "reasoning": f"Errore verifica LLM, mantenuto tipo originale",
      }
   except Exception as e:
      print(f"Error calling LLM: {e}")
      return {
         "is_valid_pii": True,
         "corrected_type": None,
         "confidence_score": entity["confidence"],
         "reasoning": f"Errore verifica: {str(e)}",
      }


def pii_detection_pipeline(text: str, verify_uncertain: bool = True) -> dict:
   """
   Pipeline completa: NER con DeBERTa + Verifica LLM locale per casi dubbi
   """
   print(f"Text analysis: {len(text)} char")

   # DeBERTa call
   confident_entities, uncertain_entities = extract_entities_with_deberta(text)

   print(f"\nConfident entities: {len(confident_entities)}")
   print(f"Uncertain entities: {len(uncertain_entities)}\n")

   verified_entities = confident_entities.copy()
   llm_verified = 0
   llm_rejected = 0

   # LLM validation for uncertain enities
   if verify_uncertain and uncertain_entities:
      print("\n - Local LLM validation")

      for i, uncertain_entity in enumerate(uncertain_entities, 1):
         print(
               f"  [{i}/{len(uncertain_entities)}] validation: {uncertain_entity['text']}"
         )

         llm_result = verify_with_llama_local(uncertain_entity, text)

         if llm_result["is_valid_pii"]:
               # Enity validated by LLM
               final_type = (
                  llm_result.get("corrected_type") or uncertain_entity["type"]
               )

               verified_entity = {
                  "text": uncertain_entity["text"],
                  "type": final_type,
                  "confidence": llm_result["confidence_score"],
                  "start": uncertain_entity["start"],
                  "end": uncertain_entity["end"],
                  "source": "llm_verified",
                  "original_type": uncertain_entity["type"],
                  "original_confidence": uncertain_entity["confidence"],
                  "llm_reasoning": llm_result["reasoning"],
               }

               verified_entities.append(verified_entity)
               llm_verified += 1
               print(f"Accepted as {final_type}")
         else:
               llm_rejected += 1
               print(f"Refused: {llm_result['reasoning']}")

   # Final stats
   results = {
      "text": text,
      "verified_entities": verified_entities,
      "statistics": {
         "total_entities": len(verified_entities),
         "confident_from_deberta": len(confident_entities),
         "uncertain_from_deberta": len(uncertain_entities),
         "llm_verified": llm_verified,
         "llm_rejected": llm_rejected,
         "final_precision_estimate": (len(confident_entities) + llm_verified)
         / (len(confident_entities) + len(uncertain_entities))
         if (len(confident_entities) + len(uncertain_entities)) > 0
         else 0,
      },
   }

   return results

def redact_text(text: str, entities: List[dict]) -> dict:
   """
   Redacts PII entities from text by replacing them with block characters (█).
   
   Args:
      text: Original text containing PII
      entities: List of entity dictionaries with 'start', 'end', 'text', and 'type' keys
   
   Returns:
      Dictionary containing:
         - redacted_text: Text with PII replaced by █ characters
         - redaction_map: List of redactions with positions and types
         - original_text: Original unredacted text
   """
   # Sort entities by start position (reverse order for safe replacement)
   sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
   
   redacted_text = text
   redaction_map = []
   
   # Replace entities from end to start to maintain correct positions
   for entity in sorted_entities:
      start = entity['start']
      end = entity['end']
      entity_text = entity['text']
      entity_type = entity['type']
      
      # Calculate number of block characters (match original length)
      redaction_length = len(entity_text)
      replacement = '█' * redaction_length
      
      # Replace in text
      redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
      
      # Track redaction
      redaction_map.append({
         'original_text': entity_text,
         'type': entity_type,
         'start': start,
         'end': end,
         'redacted_with': replacement
      })
   
   # Reverse redaction_map to show in original order
   redaction_map.reverse()
   
   return {
      'redacted_text': redacted_text,
      'redaction_map': redaction_map,
      'original_text': text,
      'total_redactions': len(redaction_map)
   }


def redact_text_compact(text: str, entities: List[dict]) -> dict:
   """
   Redacts PII with fixed-length blocks for enhanced privacy.
   All entities replaced with same length block (default: 5 characters).
   
   Args:
      text: Original text containing PII
      entities: List of entity dictionaries with 'start', 'end', 'text', and 'type' keys
   
   Returns:
      Dictionary containing redacted text and metadata
   """
   REDACTION_SYMBOL = '█████'  # Fixed length redaction
   
   sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
   redacted_text = text
   redaction_map = []
   
   for entity in sorted_entities:
      start = entity['start']
      end = entity['end']
      
      # Replace with fixed-length block
      redacted_text = redacted_text[:start] + REDACTION_SYMBOL + redacted_text[end:]
      
      redaction_map.append({
         'original_text': entity['text'],
         'type': entity['type'],
         'start': start,
         'end': start + len(REDACTION_SYMBOL),  # New end position
         'redacted_with': REDACTION_SYMBOL
      })
   
   redaction_map.reverse()
   
   return {
      'redacted_text': redacted_text,
      'redaction_map': redaction_map,
      'original_text': text,
      'total_redactions': len(redaction_map)
   }

def pii_detection_with_redaction(text: str, verify_uncertain: bool = True, 
                                 redaction_style: str = 'length_match') -> dict:
   """
   Complete pipeline: Detection + Redaction
   
   Args:
      text: Input text to analyze
      verify_uncertain: Whether to verify uncertain entities with LLM
      redaction_style: 'length_match' or 'compact'
   
   Returns:
      Dictionary with detection results and redacted text
   """
   # Run existing pipeline
   results = pii_detection_pipeline(text, verify_uncertain)
   
   # Redact text based on verified entities
   entities = results['verified_entities']
   
   if redaction_style == 'compact':
      redaction_results = redact_text_compact(text, entities)
   else:  # length_match (default)
      redaction_results = redact_text(text, entities)
   
   # Combine results
   results['redaction'] = redaction_results
   
   return results




if __name__ == "__main__":
   sample_text = """
   TICKET #CST-2024-8891 - Account Access Issue

   From: Jennifer Martinez (jennifer.martinez@techcorp.com)
   Employee ID: EMP-447823
   Date: 2024-03-15 14:32:08 EST

   Dear Support Team,

   I'm writing regarding my corporate account. My full details are:

   Personal Information:
   - Full Name: Jennifer Marie Martinez
   - Date of Birth: 07/23/1989
   - SSN: 456-88-9012
   - Driver's License: D2389456 (CA)
   - Passport: 482394856

   Contact Details:
   - Primary Phone: +1 (415) 882-3394
   - Secondary Phone: 415.882.3395
   - Home Address: 2847 Elm Street, Apt 12B, San Francisco, CA 94102
   - Username: jmartinez_techcorp
   - Temporary Password: TempP@ss2024!

   Financial Information:
   - Credit Card: 4532-8821-9034-7756
   - CVV: 842
   - Expiration: 08/2027
   - IBAN: GB29NWBK60161331926819
   - BBAN: NWBK60161331926819
   - Routing Number: 121000248
   - SWIFT/BIC: CHASUS33XXX
   - Account PIN: 7739

   System Access:
   - Customer ID: CUST_4728391
   - API Key: sk_live_51HyT8KLm3n4P5qR6sT7uV8wX9yZ0aB1cD2eF3gH4iJ5kL6mN7oP8qR9s
   - Server IP: 192.168.1.45
   - IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
   - Login Time: 09:45:22 AM

   The meeting is scheduled for March 20, 2024 at 3:30 PM PST. 

   Please contact me at work (TechCorp Solutions, office 4B) or my colleague Robert Chen 
   (robert.chen@techcorp.com, Employee ID: EMP-883921, phone: 415-882-9847).

   Best regards,
   Jennifer Martinez
   CONFIDENTIAL MEDICAL RECORD - Harbor Medical Center

   Patient: Dr. Michael James Thompson
   DOB: 12/15/1976
   SSN: 789-22-4456
   Medical Record #: MRN-2024-7834
   Insurance ID: INS-8837291

   Emergency Contact:
   Sarah Thompson (spouse)
   Phone: (617) 445-8821
   Address: 1523 Oak Boulevard, Boston, MA 02134

   Appointment History:
   - Last Visit: 2024-02-28 at 10:15:00
   - Next Scheduled: 2024-04-10 at 14:30:00

   Prescriptions can be sent to:
   CVS Pharmacy #4472
   coordinates: 42.3601° N, 71.0589° W
   Phone: 617-445-9923

   Payment Information:
   Card Number: 5425-2334-8821-4429
   Security Code: 391
   Account Holder: Michael J. Thompson

   Provider Notes:
   Dr. Emily Watson (License #MD-449823)
   Email: e.watson@harbormedical.org
   Direct Line: 617-445-7712
   Available: Mon-Fri 8:00 AM - 5:00 PM

   Referred by: Dr. James Patterson
   Contact: j.patterson@bostonhealth.com
   Office: Boston Health Associates
   ONLINE BANKING APPLICATION FORM

   Applicant Information:
   ----------------------------------------
   First Name: Alexander
   Last Name: Rodriguez
   Full Legal Name: Alexander Miguel Rodriguez
   Birth Date: 03/08/1992
   Social Security: 234-67-8890

   Current Address:
   789 Pine Street, Unit 5D
   Seattle, WA 98101
   GPS Coordinates: 47.6062° N, 122.3321° W

   Contact Information:
   Primary: +1-206-774-5521
   Alternative: 206.774.5522
   Email: alex.rodriguez92@gmail.com
   Work Email: arodriguez@innovatetech.com

   Employment Details:
   Company: InnovateTech Systems
   Employee Number: ET-992847
   Position Start Date: 01/15/2020
   Office Phone: (206) 555-0198

   Existing Accounts:
   Current Bank IBAN: DE89370400440532013000
   SWIFT Code: COBADEFFXXX
   Routing #: 125000024
   Previous Account PIN: 4892

   Identity Verification:
   Driver's License: WA-DL-8847293
   Passport Number: 667382910
   Issue Date: 06/20/2022
   Expiry: 06/19/2032

   Online Banking Setup:
   Requested Username: alex_rod_92
   Initial Password: SecureB@nk2024#
   Security Question Answer: Fluffy (pet name)

   Credit Check Authorization:
   Card on File: 6011-0009-9013-4422
   CVV: 728
   Valid Through: 11/2026

   Application Timestamp: 2024-03-18T16:42:33Z
   IP Address: 203.0.113.45
   IPv6 Address: 2001:db8:3333:4444:5555:6666:7777:8888

   References:
   1. Maria Santos (maria.santos@outlook.com, 206-774-8839)
   2. David Kim (Customer ID: CK-774829, david.kim@company.com)

   API Integration:
   Developer Key: pk_live_8aK3mN9pQ2rS5tU7vW1xY4zB6cD8eF0gH2iJ4kL6mN8oP0qR2s
   Webhook URL: https://api.innovatetech.com/banking/webhooks

   Authorization Code: AUTH-2024-MTR-7738
   Processing Date: March 18, 2024 at 4:45 PM
   HUMAN RESOURCES DATABASE EXPORT - Q1 2024
   ================================================================

   EMPLOYEE #1:
   Name: Rebecca Lynn Foster
   Employee ID: HR-2024-4482
   DOB: 09/14/1988
   SSN: 567-33-9021
   Hire Date: 01/10/2022

   Contact:
   Phone: (312) 667-8832
   Email: rfoster@globalcorp.com
   Address: 4521 Madison Avenue, Chicago, IL 60614
   Emergency: Tom Foster - 312-667-8833

   Credentials:
   - Username: rfoster.global
   - Temp Password: Welcome2024!
   - Badge ID: BADGE-48821
   - Parking Spot: P-447
   - Access Card: AC-7829311

   Financial:
   Direct Deposit: IBAN IT60X0542811101000000123456
   SWIFT: BPMIITMMXXX
   Last 4 of Card: 8821
   Tax ID: 567-33-9021

   ----------------------------------------------------------------

   EMPLOYEE #2:
   Name: Omar Hassan Al-Sayed
   First: Omar
   Last: Al-Sayed
   Employee ID: HR-2024-9983
   Born: 05/22/1995
   SSN: 893-55-7721
   Started: 03/15/2023 at 9:00 AM

   Reachable at:
   Cell: 312-774-9921
   Work Phone: (312) 555-0147 ext. 4482
   Personal: omar.alsayed@email.com
   Corporate: o.alsayed@globalcorp.com
   Living at: 8834 Lakeview Terrace, Evanston, IL 60201

   System Access:
   Login: oalsayed_corp
   Initial Pass: Corp@2023Secure
   Employee Portal ID: EP-773891
   VPN IP Assignment: 10.0.45.123
   Last Login: 2024-03-19T08:23:45-05:00

   Banking Info:
   Account IBAN: FR1420041010050500013M02606
   BIC/SWIFT: BNPAFRPPXXX
   Routing: 111000025
   PIN Code: 6638

   Identification:
   Passport: N8847392
   Driver License: IL-DL-H482-9384-7721
   Issued: 07/01/2023

   ----------------------------------------------------------------

   EMPLOYEE #3:
   Names: Chen, Li Wei
   Employee #: HR-2024-5521
   Birth: 11/30/1990
   Social Security Number: 445-88-3392
   Joined: 08/20/2021

   Phones:
   Main: +1 (415) 338-7745
   Mobile: 415.338.7746
   Email: liwei.chen@globalcorp.com

   Address:
   1247 Valencia Street, Apt 8
   San Francisco, CA 94110
   Location: 37.7749° N, 122.4194° W

   Account Details:
   Username: lchen.eng
   Password: Engineering#2021
   Customer Profile: CUST-773829
   Access Level: ADMIN-8837

   Payment:
   Card: 3782-822463-10005 (AMEX)
   Security: 1847
   IBAN: ES9121000418450200051332
   SWIFT: CAIXESBBXXX
   Bank Route: 026009593
   Account Security PIN: 9384

   Documents:
   CA Driver's License: D7738492
   US Passport: 773829401
   Valid until: 12/31/2030

   System Metadata:
   Registration IP: 172.16.254.1
   IPv6: fe80::1ff:fe23:4567:890a
   API Access Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
   Session ID: sess_4j8k2m9n3p5q7r

   Last Updated: 2024-03-20 11:23:45 UTC

   ================================================================
   CONFIDENTIAL - DO NOT DISTRIBUTE
   Generated: Wednesday, March 20, 2024 at 11:30 AM EST
   Report ID: RPT-Q1-2024-HR-8837
   EMAIL THREAD: Project Aurora - Access Request

   From: James Wilson <james.wilson@techventures.io>
   To: security@techventures.io
   Date: Thursday, March 21, 2024 2:45:33 PM
   Subject: Urgent: New Team Member Onboarding

   Hi Security Team,

   We need to onboard our new contractor ASAP. Details below:

   Contractor: Sofia Gonzalez-Martinez
   Personal Email: s.gonzalez.martinez@freelancer.com
   Phone: (213) 445-7823
   Alt Phone: 213-445-7824
   DOB: 04/17/1993
   SSN: 778-44-5521 (for tax purposes)
   Address: 5621 Sunset Blvd, Los Angeles, CA 90028

   Please create:
   - Username: sgonzalez_contractor
   - Temporary Password: Contractor!2024
   - Employee ID: CONT-2024-5538
   - Access Level: Level 3

   Payment Setup:
   IBAN: ES7620770024003102575766
   SWIFT: CAHMESMMXXX
   Backup Card: 4916-3385-2947-5510 (exp: 05/2027, CVV: 482)

   She'll be working with:
   Dr. Ahmed Khan (ahmed.khan@techventures.io, EMP-4482991, office: +1-213-555-0162)
   and Lisa Chen (username: lchen_senior, ID: EMP-7738821)

   Project server: 198.51.100.42
   IPv6: 2001:db8:85a3::8a2e:370:7334
   SSH Key: ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ...

   Meeting scheduled for: 2024-03-25 at 10:00:00 AM PST
   Conference Room: Building A, coordinates: 34.0522° N, 118.2437° W

   Approval Code: APPR-2024-MTR-4482
   Processed by: Manager Susan Taylor (staylor@techventures.io)
   Manager ID: MGR-8837291
   Direct: (213) 555-0198

   Best,
   James Wilson
   Driver's License (for building access): CA-DL-K8837492
   Passport (for international travel approval): 884729401
   API Key (for automation): sk_test_4eC39HqLyjWDarjtT1zdp7dc

   ---
   Timestamp: 2024-03-21T14:45:33-07:00
   Sent from: 192.0.2.146
   Thread ID: THR-8837-MTR-2024
   Customer Support PIN: 4729
   """
   results = pii_detection_with_redaction(sample_text, verify_uncertain=False, redaction_style='length_match')

   print("=" * 60)
   print("ORIGINAL TEXT:")
   print(results['redaction']['original_text'])
   print("\n" + "=" * 60)
   print("REDACTED TEXT:")
   print(results['redaction']['redacted_text'])
   print("\n" + "=" * 60)
   print(f"REDACTION SUMMARY: {results['redaction']['total_redactions']} entities redacted")
   print("\nREDACTION MAP:")
   for redaction in results['redaction']['redaction_map']:
      print(f"  - {redaction['type']}: '{redaction['original_text']}' → {redaction['redacted_with']}")

   # Stats
   print("Final results")
   print("=" * 60)

   print("\nStats:")
   for key, value in results["statistics"].items():
      print(f"  {key}: {value}")

   print("\nPII entities found:")
   for i, entity in enumerate(results["verified_entities"], 1):
      print(f"\n  [{i}] {entity['text']}")
      print(f"Type: {entity['type']}")
      print(f"Confidence: {entity['confidence']:.2%}")
      print(f"Source: {entity.get('source', 'deberta_confident')}")

      if entity.get("llm_reasoning"):
         print(f"LLM reason: {entity['llm_reasoning']}")
