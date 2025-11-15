from pydantic import BaseModel
from typing import Optional
from datetime import date

class TargetListModel(BaseModel):
    id: Optional[int] = None
    hcp_code: Optional[str] = None
    full_name: str
    gender: Optional[str] = None
    qualification: Optional[str] = None
    specialty: Optional[str] = None
    designation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    hospital_name: Optional[str] = None
    hospital_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    experience_years: Optional[int] = None
    influence_score: Optional[float] = None
    category: Optional[str] = None
    therapy_area: Optional[str] = None
    monthly_sales: Optional[int] = None
    yearly_sales: Optional[int] = None
    last_interaction_date: Optional[date] = None
    call_frequency: Optional[int] = None
    priority: Optional[bool] = False

    class Config:
        orm_mode = True
