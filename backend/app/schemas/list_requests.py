from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class ListRequests(BaseModel):
    request_id: Optional[int] = None
    subdomain_id: int
    requester_name: str
    request_purpose: str
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        orm_mode = True