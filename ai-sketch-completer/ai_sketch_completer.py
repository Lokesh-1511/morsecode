import cv2
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

# --- Load Hugging Face Stable Diffusion model ---
print("Loading model... this may take a minute ⏳")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print("✅ Model loaded successfully!")

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open webcam!")

print("\n📸 Press 'g' to generate artwork, 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed!")
        break

    # --- Step 1: Detect sketch edges ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    inv = cv2.bitwise_not(edges)

    cv2.imshow("Your Sketch (Press 'g' to Generate)", inv)

    # --- Step 2: Key actions ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        print("🎨 Generating AI artwork... please wait.")

        # Save current sketch
        Image.fromarray(inv).save("sketch.png")
        init_image = Image.open("sketch.png").convert("RGB")

        # --- AI generation ---
        prompt = "complete this pencil sketch into a realistic, colorful painting"
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.7,
            guidance_scale=7.5
        ).images[0]

        # Show result
        result.save("result.png")
        print("✅ Artwork generated! Opening result...")
        result.show()

    elif key == ord('q'):
        print("👋 Exiting program.")
        break

cap.release()
cv2.destroyAllWindows()
