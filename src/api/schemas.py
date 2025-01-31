from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Category of the merchant")
    merchant_country: str = Field(..., description="Country where transaction occurred")
    card_type: str = Field(..., description="Type of card used")
    transaction_type: str = Field(..., description="Type of transaction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")

class TransactionCreate(TransactionBase):
    pass

class TransactionPredict(TransactionBase):
    pass

class TransactionInDB(TransactionBase):
    id: int
    prediction: Optional[bool] = None
    prediction_probability: Optional[float] = None
    prediction_timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    transaction_id: int
    is_fraudulent: bool
    confidence_score: float
    prediction_timestamp: datetime
    model_version: str
    explanation: Optional[dict] = None 