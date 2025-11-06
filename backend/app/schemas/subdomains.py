from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class Subdomains(BaseModel):
    subdomain_id: Optional[int] = None
    domain_id: int
    subdomain_name: str

    class Config:
        orm_mode = True