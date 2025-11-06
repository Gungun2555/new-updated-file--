from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class DomainsModel(BaseModel):
    domain_id: Optional[int] = None
    domain_name: str
    created_at: Optional[datetime] = None