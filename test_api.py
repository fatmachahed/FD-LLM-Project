import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

print("KEY:", key)
print("LEN:", len(key) if key else "None")






