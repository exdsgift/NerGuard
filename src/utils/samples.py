"""
Sample PII texts for testing and debugging.

This module contains example texts with various types of PII for testing
the NerGuard model and pipeline.
"""

# Comprehensive sample with multiple PII types
SAMPLE_REPORT = """REPORT HEADER
   Date: November 14, 2024
   Time: 09:45 AM
   Case ID: 88291-X
   Investigator: Mr. Jonathan Doe

   SUBJECT PROFILE
   Full Name: Dr. Alexander James Sterling
   Title: Dr.
   Given Name: Alexander
   Surname: Sterling
   Gender: Male
   Sex: M
   Age: 42

   CONTACT & RESIDENCE
   The subject currently resides at Building 404, Maplewood Avenue, in the city of Springfield, 62704. He previously lived in Chicago, Zip Code 60614.
   Current Address: 404 Maplewood Avenue, Springfield.
   Primary Email: alex.sterling.42@example-corp.net
   Secondary Email: a.sterling@private-mail.org
   Home Phone: (217) 555-0199
   Mobile Phone: +1-555-010-9988

   IDENTIFICATION DOCUMENTS
   During the verification process on October 12, 2023, the subject presented multiple forms of identification to clear the security check:
   1. Passport: His US Passport number is P44992211, expiring in 2029.
   2. Driver's License: He holds a valid license from Illinois, number S9988-7766-5544, class D.
   3. Social Security: His SSN on file is 555-01-4433.
   4. National ID / Employee ID: He uses the internal ID Card Number EMP-99283 for building access.
   5. Tax ID: For his consulting business, he uses Tax Identification Number (TIN) 99-1234567.

   FINANCIAL INCIDENT DETAILS
   On Monday, November 11, at approximately 14:30, the subject attempted a transaction that was flagged by our automated systems. The transaction involved Credit Card Number 4532 8811 9000 7654 (Visa Gold).
   The transaction took place at a merchant terminal located at 15 State Street, Boston. The subject allegedly attempted to purchase luxury goods valued at $15,000. When asked for secondary verification, he provided a Credit Card 5400 1234 5678 9010 which did not match the name on the ID.

   WITNESS STATEMENT
   Witness: Mrs. Sarah O'Connor (Manager)
   "I spoke to Mr. Sterling at 3:15 PM. He seemed agitated. He claimed he was 42 years old and had been a customer since 2015. He showed me his Passport P44992211 again, but the photo looked different. I asked him to verify his billing Zip Code, and he said 02108, which matches the Boston area, not his Springfield residence."

   CONCLUSION
   Pending further review of Social Security Number 555-01-4433 and cross-reference with Driver's License S9988-7766-5544.
"""

# Simple sample for quick testing
SAMPLE_SIMPLE = """
Dear John Smith,

Thank you for contacting us regarding your account. We have verified your identity using your email john.smith@example.com and phone number +1-555-123-4567.

Your account at 123 Main Street, New York, NY 10001 has been updated successfully.

Best regards,
Customer Service Team
"""

# Multilingual sample (Italian)
SAMPLE_ITALIAN = """
Gentile Sig. Mario Rossi,

La informiamo che la sua richiesta è stata processata. I dati verificati sono:
- Nome: Mario
- Cognome: Rossi
- Email: mario.rossi@esempio.it
- Telefono: +39 02 1234567
- Indirizzo: Via Roma 42, 20100 Milano

Codice fiscale: RSSMRA80A01H501U
Carta di credito: 4111 1111 1111 1111

Cordiali saluti,
Servizio Clienti
"""

# Edge cases sample
SAMPLE_EDGE_CASES = """
Contact info mixed in text: You can reach me at test@email.com or call 555-0100.
Numbers that aren't PII: The temperature is 72 degrees and the year is 2024.
Partial names: Dr. Smith mentioned that Prof. Johnson would attend.
Ambiguous locations: I visited the Springfield museum last summer.
Credit card like numbers: Order #4532-1111-2222-3333 (not a real card).
"""

# For backwards compatibility
SAMPLE1 = SAMPLE_REPORT


def get_sample(name: str = "report") -> str:
    """
    Get a sample text by name.

    Args:
        name: Sample name ("report", "simple", "italian", "edge_cases")

    Returns:
        Sample text string
    """
    samples = {
        "report": SAMPLE_REPORT,
        "simple": SAMPLE_SIMPLE,
        "italian": SAMPLE_ITALIAN,
        "edge_cases": SAMPLE_EDGE_CASES,
    }
    return samples.get(name.lower(), SAMPLE_REPORT)


def list_samples() -> list:
    """
    List available sample names.

    Returns:
        List of sample names
    """
    return ["report", "simple", "italian", "edge_cases"]
