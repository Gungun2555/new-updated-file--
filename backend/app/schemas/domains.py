from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class Domains(BaseModel):
    domain_id: Optional[int] = None
    domain_name: str
    created_at: Optional[datetime] = None

    class Config:
        orm_mode = True