import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
import pyopencl as cl

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GPUImageProcessor:
    def __init__(self):
        try:
            platforms = cl.get_platforms()
            self.ctx = cl.Context(
                dev_type=cl.device_type.GPU,
                properties=[(cl.context_properties.PLATFORM, platforms[0])]
            )
            self.queue = cl.CommandQueue(self.ctx)

            self.kernel_source = """
            __kernel void enhance_image(
                __global const uchar *input,
                __global uchar *output,
                const int width,
                const int height
            ) {
                int gid_x = get_global_id(0);
                int gid_y = get_global_id(1);
                
                if (gid_x >= width || gid_y >= height) {
                    return;
                }
                
                int idx = gid_y * width + gid_x;
                float pixel_value = (float)input[idx];
                
                float enhanced = pixel_value * 1.2f;
                output[idx] = (uchar)clamp(enhanced, 0.0f, 255.0f);
            }
            """
            
            self.program = cl.Program(self.ctx, self.kernel_source).build()
            self.gpu_available = True
            
        except Exception as e:
            print(f"GPU initialization failed: {str(e)}")
            print("Falling back to CPU processing...")
            self.gpu_available = False

    def enhance_image(self, image):
        if not self.gpu_available:
            return cv2.convertScaleAbs(image, alpha=1.2, beta=0)

        try:
            image = np.ascontiguousarray(image)
            
            input_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=image
            )
            output_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.WRITE_ONLY,
                image.nbytes
            )

            height, width = image.shape
            global_size = (width, height)
            local_size = None

            self.program.enhance_image(
                self.queue,
                global_size,
                local_size,
                input_buf,
                output_buf,
                np.int32(width),
                np.int32(height)
            )

            output = np.empty_like(image)
            cl.enqueue_copy(self.queue, output, output_buf)
            return output

        except Exception as e:
            print(f"GPU processing failed: {str(e)}")
            print("Falling back to CPU processing...")
            return cv2.convertScaleAbs(image, alpha=1.2, beta=0)


def increase_contrast_and_sharpen(image):
    alpha = 1.5
    beta = 20
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    sharpened = cv2.GaussianBlur(adjusted, (0, 0), 3)
    sharpened = cv2.addWeighted(adjusted, 1.5, sharpened, -0.5, 0)
    return sharpened

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def improve_image_quality(image):
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = cv2.fastNlMeansDenoising(gray)
        
        gpu_processor = GPUImageProcessor()
        
        enhanced = gpu_processor.enhance_image(denoised)
        
        enhanced = increase_contrast_and_sharpen(enhanced)
        deskewed = deskew(enhanced)

        binary = cv2.adaptiveThreshold(
            deskewed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        kernel = np.ones((1, 1), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morphed

    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return gray


def extract_text(image_path, langs="eng"):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from '{image_path}'.")

        processed_image = improve_image_quality(image)
        
        debug_image_path = "debug_processed.png"
        cv2.imwrite(debug_image_path, processed_image)
        print(f"Debug image saved to {debug_image_path}")
        
        custom_config = '--oem 1 --psm 3'
        
        pil_image = Image.fromarray(processed_image)
        text = pytesseract.image_to_string(
            pil_image,
            lang=langs,
            config=custom_config
        )
        
        return text.strip()

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return ""
    except ValueError as val_error:
        print(val_error)
        return ""
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        return ""


if __name__ == "__main__":
    image_path = r"C:\Users\MuraliDharan S\OneDrive\Desktop\monika R resume_page-0001.jpg"
    extracted_text = extract_text(image_path)
    if extracted_text:
        print("\nExtracted Text:\n", extracted_text)
    else:
        print("No text was extracted.")
