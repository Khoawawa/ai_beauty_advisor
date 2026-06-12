import requests

# Point to your raw image endpoint
url = "http://localhost:8000/api/v1/parse-face"
image_path = "test_face.jpg"
output_filename = "result_mask.png"

print(f"🚀 Sending '{image_path}' to the AI Beauty API...")

try:
    # 1. Send the photo
    with open(image_path, "rb") as file:
        files = {"file": (image_path, file, "image/jpeg")}
        response = requests.post(url, files=files)

    # 2. Check if it succeeded
    if response.status_code == 200:
        # DO NOT use response.json()
        # Instead, write the raw binary content directly to a file
        print(response.content)
        with open(output_filename, "wb") as out_file:
            out_file.write(response.content)
            
        print(f"\n✅ SUCCESS! The server returned an image.")
        print(f"🖼️ Saved the mask as '{output_filename}' in your folder!")

    else:
        print(f"\n❌ SERVER ERROR {response.status_code}:")
        print(response.text)

except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{image_path}'.")
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Connection refused. Is your FastAPI server running?")