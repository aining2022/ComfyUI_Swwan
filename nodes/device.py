# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import ImageGrab, cv2, np, time, torch

class ImageGrabPIL:

    @classmethod
    def IS_CHANGED(cls):

        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screencap"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Captures an area specified by screen coordinates.
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
                 "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
        },
    }

    def screencap(self, x, y, width, height, num_frames, delay):
        start_time = time.time()
        captures = []
        bbox = (x, y, x + width, y + height)

        for _ in range(num_frames):
            # Capture screen
            screen_capture = ImageGrab.grab(bbox=bbox)
            screen_capture_torch = torch.from_numpy(np.array(screen_capture, dtype=np.float32) / 255.0).unsqueeze(0)
            captures.append(screen_capture_torch)

            # Wait for a short delay if more than one frame is to be captured
            if num_frames > 1:
                time.sleep(delay)

        elapsed_time = time.time() - start_time
        print(f"screengrab took {elapsed_time} seconds.")

        return (torch.cat(captures, dim=0),)

class WebcamCaptureCV2:

    @classmethod
    def IS_CHANGED(cls):
        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "capture"
    CATEGORY = "Swwan/experimental"
    DESCRIPTION = """
Captures a frame from a webcam using CV2.
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "cam_index": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "release": ("BOOLEAN", {"default": False}),
            },
        }

    def capture(self, x, y, cam_index, width, height, release):
        # Check if the camera index has changed or the capture object doesn't exist
        if not hasattr(self, "cap") or self.cap is None or self.current_cam_index != cam_index:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
            self.current_cam_index = cam_index
            self.cap = cv2.VideoCapture(cam_index)
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except:
                pass
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")

        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from webcam")

        # Crop the frame to the specified bbox
        frame = frame[y:y+height, x:x+width]
        img_torch = torch.from_numpy(frame[..., [2, 1, 0]]).float() / 255.0

        if release:
            self.cap.release()
            self.cap = None

        return (img_torch.unsqueeze(0),)
