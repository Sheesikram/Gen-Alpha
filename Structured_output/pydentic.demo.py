from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import json

class Student(BaseModel):
    name: str = 'Salu'
    age: Optional[int] = 0
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

# Create a dictionary instead of a set for proper key-value mapping
new_s = {
    "name": "salman",
    "age": 20,
    "email": "shees@gmail.com",
    "cgpa": 5.0
}
new_s2={"email": "shees2@gmail.com"}
# Create student instance
student = Student(**new_s2)# ** operator in Python is used for dictionary unpacking

print("Student Object:")
print(student)

# Convert to dictionary
student_dict = student.model_dump()
print("\nStudent Dictionary:")
print(student_dict)

# Convert to JSON string
student_json = student.model_dump_json()
print("\nStudent JSON:")
print(student_json)

# Save to JSON file
with open("student_data.json", "w") as f:
    json.dump(student_dict, f, indent=4)

print("\nJSON file has been created: student_data.json")

