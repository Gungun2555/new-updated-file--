from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class SubdomainsModel(BaseModel):
    subdomain_id: Optional[int] = None
    domain_id: int
    subdomain_name: str