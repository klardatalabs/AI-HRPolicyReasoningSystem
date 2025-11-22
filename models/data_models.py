from typing import Optional
from pydantic import BaseModel, EmailStr


class User(BaseModel):
    email_id: EmailStr
    password: str
    hashed_password: Optional[str] | None = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserEmail(BaseModel):
    user_email_id: EmailStr