import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from PIL import Image

import google.generativeai as genai

class GeminiAPI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="You are an invoice parser. Extract date and financial transactions including item, amount and currency. If no currency is specified, assume VND. If the amount is not clear, indicate 'N/A'. Always provide a JSON output with format '{ 'transactions' : {'<date>' : [<list of transaction objects>]}}' and a 'raw_text' key with the OCR text."
        )

    def parse_invoice_image(self, image_bytes):
        print("Sending image to Gemini for parsing...")
        # Create a GenerativeModel and start a chat
        image_part = {
            "mime_type": "image/jpeg", # Assuming JPEG for invoice images
            "data": image_bytes
        }
        prompt = "Extract financial transactions from this invoice."
        response = self.model.generate_content([prompt, image_part], stream=False)

        # Extract the relevant content. Assuming the response will contain JSON in text.
        # You might need to refine this based on actual Gemini API response structure.
        try:
            import json
            content = response.text.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini response: {e}")
            print(f"Raw Gemini response: {response.text}")
            return {"transactions": [], "raw_text": response.text}
        except Exception as e:
            print(f"An unexpected error occurred during Gemini parsing: {e}")
            return {"transactions": [], "raw_text": response.text}


def authenticate_google_drive(service_account_file):
    # Replace with your actual Google Drive API authentication
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def download_invoice_image(drive_service, file_id):
    print(f"Downloading file with ID: {file_id}")
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.seek(0)
    return Image.open(fh)

def main():
    # --- Configuration ---
    SERVICE_ACCOUNT_FILE = f"{os.getenv('SERVICE_ACCOUNT_FILE')}"
    GEMINI_API_KEY = f"{os.getenv('GEMINI_API_KEY')}"
    GOOGLE_DRIVE_FOLDER_ID = f"{os.getenv('GOOGLE_DRIVE_FOLDER_ID')}"

    # --- Google Drive Authentication ---
    try:
        drive_service = authenticate_google_drive(SERVICE_ACCOUNT_FILE)
        print("Successfully authenticated with Google Drive.")
    except Exception as e:
        print(f"Error authenticating with Google Drive: {e}")
        return

    # --- Gemini API Initialization ---
    gemini_api = GeminiAPI(GEMINI_API_KEY)

    # --- Process Invoices ---
    print(f"Listing files in Google Drive folder: {GOOGLE_DRIVE_FOLDER_ID}")
    try:
        results = drive_service.files().list(
            q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents",
            fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found in the specified Google Drive folder.')
        else:
            print('Processing invoices:')
            for item in items:
                file_id = item['id']
                file_name = item['name']
                print(f"\nProcessing invoice: {file_name} (ID: {file_id})")

                try:
                    image = download_invoice_image(drive_service, file_id)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    img_byte_arr = img_byte_arr.getvalue()

                    # Send to Gemini for parsing
                    parsed_data = gemini_api.parse_invoice_image(img_byte_arr)
                    print(f"Parsed Transactions for {file_name}")
                    print(f"Raw OCR Text for {file_name}: {parsed_data['raw_text']}")
                    for date, items in  parsed_data["transactions"].items():
                        print(date)
                        for item in items:
                            print(f"{item["item"]}:{item["amount"]} {item["currency"]}")                 
                except Exception as e:
                    print(f"Error processing file {file_name} (ID: {file_id}): {e}")

    except Exception as e:
        print(f"Error listing files in Google Drive folder: {e}")

if __name__ == '__main__':
    main()
