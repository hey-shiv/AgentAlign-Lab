from pydantic import BaseModel


class Student(BaseModel):
    name: str
    age: int


s = Student(name="Shivashant", age="hello")

print(s)