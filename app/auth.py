"""
Authentication utilities for JWT token handling.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.database import db_manager
from app.models import Pedagang
from app.schemas import TokenData


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode a JWT access token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        pedagang_id: int = payload.get("pedagang_id")
        
        if username is None:
            return None
        
        return TokenData(username=username, pedagang_id=pedagang_id)
    except JWTError:
        return None


async def get_current_pedagang(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(db_manager.get_db)
) -> Pedagang:
    """
    Get the current authenticated pedagang from the JWT token.
    
    Args:
        token: JWT token from request header
        db: Database session
        
    Returns:
        Pedagang object if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token tidak valid atau sudah kadaluarsa",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_access_token(token)
    if token_data is None:
        raise credentials_exception
    
    pedagang = db_manager.get_pedagang_by_username(db, token_data.username)
    if pedagang is None:
        raise credentials_exception
    
    return pedagang


def authenticate_pedagang(db: Session, username: str, password: str) -> Optional[Pedagang]:
    """
    Authenticate a pedagang with username and password.
    
    Args:
        db: Database session
        username: Pedagang username
        password: Plain text password
        
    Returns:
        Pedagang object if authentication succeeds, None otherwise
    """
    pedagang = db_manager.get_pedagang_by_username(db, username)
    
    if not pedagang:
        return None
    
    if not verify_password(password, pedagang.password):
        return None
    
    return pedagang
