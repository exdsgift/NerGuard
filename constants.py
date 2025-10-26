
CONFIDENCE_THRESHOLD = 0.50  # * Modify this!

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