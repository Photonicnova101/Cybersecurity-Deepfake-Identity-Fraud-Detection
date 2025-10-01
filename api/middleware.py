"""
Middleware for API authentication, rate limiting, and logging
"""

import os
import time
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis)
request_counts = defaultdict(list)

# API keys (in production, use database)
VALID_API_KEYS = set(os.getenv("API_KEYS", "test-api-key-123").split(","))


def setup_middleware(app: FastAPI):
    """Setup all middleware for the app"""
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests"""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.2f}s"
        )
        
        return response
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers"""
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """
    Verify API key from Authorization header
    
    Args:
        authorization: Authorization header value
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException if invalid
    """
    # Check if API authentication is required
    require_auth = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
    
    if not require_auth:
        return "development"
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    # Extract API key (format: "Bearer <api_key>")
    try:
        scheme, api_key = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use: Bearer <api_key>"
        )
    
    # Validate API key
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


def rate_limit(request: Request, api_key: str = Header(None)):
    """
    Rate limiting middleware
    
    Args:
        request: FastAPI request
        api_key: API key from header
        
    Raises:
        HTTPException if rate limit exceeded
    """
    # Get rate limit from config
    rate_limit_per_hour = int(os.getenv("RATE_LIMIT", 100))
    
    # Use IP address as identifier
    client_ip = request.client.host
    identifier = f"{client_ip}:{api_key}" if api_key else client_ip
    
    current_time = time.time()
    hour_ago = current_time - 3600
    
    # Clean old requests
    request_counts[identifier] = [
        req_time for req_time in request_counts[identifier]
        if req_time > hour_ago
    ]
    
    # Check rate limit
    if len(request_counts[identifier]) >= rate_limit_per_hour:
        logger.warning(f"Rate limit exceeded for {identifier}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {rate_limit_per_hour} requests per hour."
        )
    
    # Add current request
    request_counts[identifier].append(current_time)


class RateLimiter:
    """
    Advanced rate limiter with configurable limits
    """
    
    def __init__(self, requests_per_minute: int = 10, requests_per_hour: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_counts = defaultdict(list)
        self.hour_counts = defaultdict(list)
    
    def check_limit(self, identifier: str) -> tuple[bool, str]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Client identifier
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        # Clean old requests
        self.minute_counts[identifier] = [
            t for t in self.minute_counts[identifier] if t > minute_ago
        ]
        self.hour_counts[identifier] = [
            t for t in self.hour_counts[identifier] if t > hour_ago
        ]
        
        # Check minute limit
        if len(self.minute_counts[identifier]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
        
        # Check hour limit
        if len(self.hour_counts[identifier]) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
        
        # Add current request
        self.minute_counts[identifier].append(current_time)
        self.hour_counts[identifier].append(current_time)
        
        return True, ""


# Global rate limiter instance
rate_limiter = RateLimiter()
