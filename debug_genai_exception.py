
import os
import sys

print(f"Python executable: {sys.executable}")

try:
    from google import genai
    print("Successfully imported google.genai")
except ImportError:
    print("Failed to import google.genai")
    sys.exit(1)

try:
    from google.api_core import exceptions as google_exceptions
    print("Successfully imported google.api_core.exceptions")
    HAS_API_CORE = True
except ImportError:
    print("Failed to import google.api_core.exceptions")
    HAS_API_CORE = False

if HAS_API_CORE:
    RETRY_EXCEPTIONS = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.TooManyRequests,
        google_exceptions.InternalServerError,
        google_exceptions.ResourceExhausted,
        google_exceptions.Aborted,
        ConnectionError,
        ConnectionRefusedError,
        TimeoutError,
        OSError,
    )
else:
    RETRY_EXCEPTIONS = (Exception,)

print(f"RETRY_EXCEPTIONS defined: {RETRY_EXCEPTIONS}")

def test_genai_exception():
    print("\n--- Testing Gemini Error Handling ---")
    
    # Use a fake key to force authentication error (401) or InvalidArgument
    # If 401 is NOT in the retry list, isinstance should be False.
    # If we want to test if it IS a google_exception, we'll see.
    fake_key = "AIzaSyFakeKeyForTestingExceptionHandling123"
    
    client = genai.Client(api_key=fake_key)
    
    try:
        print("Attempting to generate content with fake key...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello, this is a test.",
        )
        print("Unexpectedly succeeded?!")
    except Exception as e:
        print(f"\nCaught exception: {type(e).__name__}")
        print(f"Exception details: {e}")
        print(f"Type(e): {type(e)}")
        
        is_retryable = isinstance(e, RETRY_EXCEPTIONS)
        print(f"Is instance of RETRY_EXCEPTIONS? {is_retryable}")
        
        if HAS_API_CORE:
            # Check if it inherits from GoogleAPICallError which is the base for most api_core exceptions
            is_google_api_error = isinstance(e, google_exceptions.GoogleAPICallError)
            print(f"Is instance of google.api_core.exceptions.GoogleAPICallError? {is_google_api_error}")

if __name__ == "__main__":
    test_genai_exception()
