import os
import cv2
import numpy as np
import torch
from rembg import remove
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
from transformers import CLIPProcessor, CLIPModel

class ProductImageIntegrator:
    def __init__(self, device="cpu"):
        self.device = device
        # Initialize models
        self.load_models()
        
    def load_models(self):
        # Background removal is handled by rembg
        
        # Load CLIP for scene understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load stable diffusion with controlnet for inpainting
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        self.inpaint_pipeline.to(self.device)
    
    def remove_background(self, product_image):
        """Remove background from product image"""
        return remove(product_image)
    
    def analyze_scene(self, background_image):
        """Analyze background to find suitable placement locations"""
    
        # Convert PIL Image to numpy array if needed
        if isinstance(background_image, Image.Image):
            bg_np = np.array(background_image)
        else:
            bg_np = background_image.copy()
        
        # Convert to RGB if it's BGR (OpenCV default)
        if bg_np.shape[2] == 3:
            bg_rgb = cv2.cvtColor(bg_np, cv2.COLOR_BGR2RGB) if len(bg_np.shape) == 3 else bg_np
        else:
            bg_rgb = bg_np
            
        # Create PIL Image for CLIP
        bg_pil = Image.fromarray(bg_rgb)
        
        # Surface types to look for
        placement_surfaces = [
            "a table", "a desk", "a shelf", "a counter", "a floor",
            "a coffee table", "a side table", "a nightstand", "a cabinet",
            "a windowsill", "a kitchen counter", "a dining table", "a console table"
        ]
        
        # Use CLIP to identify likely surfaces
        inputs = self.clip_processor(
            text=placement_surfaces, 
            images=bg_pil, 
            return_tensors="pt", 
            padding=True
        )
        outputs = self.clip_model(**inputs)
        
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Find surfaces with highest probabilities
        top_surface_indices = torch.topk(probs[0], k=3).indices
        top_surfaces = [placement_surfaces[idx] for idx in top_surface_indices]
        
        # Now we'll use OpenCV to segment potential placement areas
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = bg_np.shape[0] * bg_np.shape[1] * 0.01  # 1% of image area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Identify horizontal surfaces using edge analysis
        placement_areas = []
        for contour in large_contours:
            # Approximate contour to simplify
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Check for horizontal line (potential surface)
            hull = cv2.convexHull(contour)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Extract angle of the rectangle
            angle = rect[2]
            
            # If angle is close to 0 or 90, it might be a horizontal surface
            if abs(angle) < 10 or abs(angle - 90) < 10:
                # Additional check for surface-like properties
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < 5:  # Not too elongated
                    # Create a mask for this contour
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    # Calculate position features
                    center_y = y + h/2
                    rel_height = center_y / bg_np.shape[0]
                    
                    # Surfaces like tables tend to be in the middle-lower part of the image
                    surface_score = 1.0
                    if rel_height < 0.3:
                        surface_score *= 0.5  # Penalize areas too high in the image
                    elif rel_height > 0.85:
                        surface_score *= 0.7  # Slightly penalize areas at the very bottom
                    
                    # Calculate additional features like texture homogeneity could be added here
                    
                    placement_areas.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': w * h,
                        'surface_score': surface_score,
                        'likely_surface_type': top_surfaces[0]
                    })
        
        # If no horizontal surfaces found using contour analysis, fall back to simple segmentation
        if not placement_areas:
            # Divide image into horizontal thirds
            h, w = bg_np.shape[:2]
            
            # Middle third often contains tables, shelves, etc.
            middle_third = {'bbox': (0, h//3, w, h//3), 'center': (w//2, h//2), 'area': w * h//3,
                            'surface_score': 0.8, 'likely_surface_type': top_surfaces[0]}
            
            # Bottom third often contains floors
            bottom_third = {'bbox': (0, 2*h//3, w, h//3), 'center': (w//2, 5*h//6), 'area': w * h//3,
                            'surface_score': 0.7, 'likely_surface_type': "a floor"}
            
            placement_areas = [middle_third, bottom_third]
        
        # Sort placement areas by score
        placement_areas.sort(key=lambda x: x['surface_score'], reverse=True)
        
        # Add semantic information from CLIP
        for area in placement_areas:
            # Determine if this area matches the surface types detected by CLIP
            if any(surface.lower() in area['likely_surface_type'].lower() for surface in top_surfaces):
                area['surface_score'] *= 1.2  # Boost score if it matches CLIP's prediction
        
        # Return the top 3 placement areas
        return placement_areas[:3]
        
    
    def optimize_placement(self, product_no_bg, background, placement_area):
        """Determine optimal size, position and perspective for placing product in background"""
        # Convert images to numpy arrays if they're PIL Images
        if isinstance(product_no_bg, Image.Image):
            product_np = np.array(product_no_bg)
        else:
            product_np = product_no_bg.copy()
        
        if isinstance(background, Image.Image):
            bg_np = np.array(background)
            bg_height, bg_width = bg_np.shape[:2]
        else:
            bg_np = background.copy()
            bg_height, bg_width = bg_np.shape[:2]
        
        # If placement_area is None, use default placement
        if placement_area is None:
            # Default to bottom center of the image
            placement_area = {
                'bbox': (bg_width // 4, bg_height // 2, bg_width // 2, bg_height // 2),
                'center': (bg_width // 2, 3 * bg_height // 4)
            }
        
        # Extract product dimensions
        prod_height, prod_width = product_np.shape[:2]
        
        # Create alpha mask for the product
        if product_np.shape[2] == 4:  # RGBA
            alpha_mask = product_np[:, :, 3] > 0
            # Create a 3-channel RGB image
            product_rgb = product_np[:, :, :3].copy()
        else:  # RGB
            # Assume all non-black pixels are part of the product
            alpha_mask = np.any(product_np > 10, axis=2)
            product_rgb = product_np.copy()
        
        # Calculate bounding box of the product without background
        y_indices, x_indices = np.where(alpha_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # If no valid pixels, return original product
            position = (0, 0)
            return product_np, alpha_mask, position
        
        top, bottom = y_indices.min(), y_indices.max()
        left, right = x_indices.min(), x_indices.max()
        
        # Crop product to its bounding box
        product_cropped = product_rgb[top:bottom+1, left:right+1]
        alpha_cropped = alpha_mask[top:bottom+1, left:right+1]
        
        # Determine appropriate size for the product based on the placement area
        placement_x, placement_y, placement_w, placement_h = placement_area['bbox']
        
        # Calculate scaling factor
        # We'll aim for the product to occupy about 30-50% of the placement area width
        target_width_ratio = np.random.uniform(0.3, 0.5)
        target_width = int(placement_w * target_width_ratio)
        
        # Calculate scaling factor
        scale_factor = target_width / (right - left + 1)
        
        # Ensure the product doesn't get too large
        max_height = int(placement_h * 0.8)  # Product shouldn't be taller than 80% of placement area
        if int((bottom - top + 1) * scale_factor) > max_height:
            scale_factor = max_height / (bottom - top + 1)
        
        # Resize product and mask
        new_width = int((right - left + 1) * scale_factor)
        new_height = int((bottom - top + 1) * scale_factor)
        
        product_resized = cv2.resize(product_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha_cropped.astype(np.uint8) * 255, (new_width, new_height), 
                                interpolation=cv2.INTER_AREA) > 127
        
        # Apply perspective transformation based on placement area
        # For simplicity, we'll use a basic perspective approximation
        
        # First, determine if the placement area suggests a perspective transformation
        if placement_area.get('likely_surface_type', '').lower() in ['a table', 'a desk', 'a coffee table', 'a dining table']:
            # Apply perspective transformation for table-like surfaces
            
            # Determine the perspective angle based on the placement in the image
            # Items higher in the image typically need more perspective
            rel_y_pos = placement_area['center'][1] / bg_height
            perspective_strength = 0.2 - 0.2 * rel_y_pos  # Stronger at top, weaker at bottom
            
            # Define source points (corners of the product)
            src_pts = np.array([
                [0, 0],
                [new_width - 1, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1]
            ], dtype=np.float32)
            
            # Define destination points with perspective
            # We'll make the bottom wider than the top for a "looking down" effect
            horizontal_shift = int(new_width * perspective_strength)
            dst_pts = np.array([
                [horizontal_shift, 0],
                [new_width - 1 - horizontal_shift, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply perspective transform
            product_perspective = cv2.warpPerspective(product_resized, perspective_matrix, 
                                                    (new_width, new_height))
            alpha_perspective = cv2.warpPerspective(alpha_resized.astype(np.uint8) * 255, 
                                                    perspective_matrix, 
                                                    (new_width, new_height)) > 127
            
            # Update product and mask
            product_transformed = product_perspective
            alpha_transformed = alpha_perspective
        else:
            # No perspective transformation for other surfaces
            product_transformed = product_resized
            alpha_transformed = alpha_resized
        
        # Determine final position within the placement area
        # We'll center it horizontally and place it at the bottom of the area
        x_pos = placement_area['center'][0] - new_width // 2
        y_pos = placement_y + placement_h - new_height
        
        # Ensure the object is within the image bounds
        x_pos = max(0, min(x_pos, bg_width - new_width))
        y_pos = max(0, min(y_pos, bg_height - new_height))
        
        position = (x_pos, y_pos)
        
        # Create final image with transparent background
        transformed_product = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        transformed_product[:, :, :3] = product_transformed
        transformed_product[:, :, 3] = alpha_transformed.astype(np.uint8) * 255
        
        return transformed_product, alpha_transformed, position
    
    def adjust_lighting(self, product, background, position):
        """Match product lighting with background scene"""
        # Convert images to numpy arrays if they're PIL Images
        if isinstance(product, Image.Image):
            product_np = np.array(product)
        else:
            product_np = product.copy()
        
        if isinstance(background, Image.Image):
            bg_np = np.array(background)
        else:
            bg_np = background.copy()
        
        # Extract product dimensions
        prod_height, prod_width = product_np.shape[:2]
        bg_height, bg_width = bg_np.shape[:2]
        
        # Create a mask for the product
        if product_np.shape[2] == 4:  # RGBA
            product_mask = product_np[:, :, 3] > 0
            product_rgb = product_np[:, :, :3].copy()
        else:
            product_mask = np.ones((prod_height, prod_width), dtype=bool)
            product_rgb = product_np.copy()
        
        # Position coordinates
        x_pos, y_pos = position
        
        # Calculate the region of interest in the background
        # We'll analyze a larger area around the product placement
        roi_x_start = max(0, x_pos - prod_width//2)
        roi_y_start = max(0, y_pos - prod_height//2)
        roi_x_end = min(bg_width, x_pos + prod_width + prod_width//2)
        roi_y_end = min(bg_height, y_pos + prod_height + prod_height//2)
        
        # Extract the region of interest from the background
        bg_roi = bg_np[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # If ROI is too small, extend to whole background
        if bg_roi.size == 0:
            bg_roi = bg_np
        
        # Convert to HSV for better color analysis
        bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_RGB2HSV)
        product_hsv = cv2.cvtColor(product_rgb, cv2.COLOR_RGB2HSV)
        
        # 1. Analyze background lighting
        # Calculate average values for background
        bg_avg_h = np.mean(bg_hsv[:, :, 0])
        bg_avg_s = np.mean(bg_hsv[:, :, 1])
        bg_avg_v = np.mean(bg_hsv[:, :, 2])
        
        # Calculate standard deviation for variance in lighting
        bg_std_v = np.std(bg_hsv[:, :, 2])
        
        # 2. Extract product lighting characteristics
        # Calculate average values for product
        product_avg_h = np.mean(product_hsv[:, :, 0][product_mask])
        product_avg_s = np.mean(product_hsv[:, :, 1][product_mask])
        product_avg_v = np.mean(product_hsv[:, :, 2][product_mask])
        
        # 3. Determine lighting direction
        # Create a gradient map of the background value channel
        bg_v = bg_hsv[:, :, 2]
        grad_x = cv2.Sobel(bg_v, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(bg_v, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the average gradient direction
        avg_grad_x = np.mean(grad_x)
        avg_grad_y = np.mean(grad_y)
        
        light_direction = np.arctan2(avg_grad_y, avg_grad_x)
        light_strength = np.sqrt(avg_grad_x**2 + avg_grad_y**2)
        
        # 4. Apply lighting adjustments
        # Create adjustment factors
        # We'll shift the product's HSV values towards the background's
        h_shift = (bg_avg_h - product_avg_h) * 0.5  # Hue shift (partial)
        s_shift = (bg_avg_s - product_avg_s) * 0.7  # Saturation shift (stronger)
        v_shift = (bg_avg_v - product_avg_v) * 0.5  # Value shift (partial)
        
        # Apply global adjustments
        adjusted_hsv = product_hsv.copy()
        adjusted_hsv[:, :, 0] = np.clip(adjusted_hsv[:, :, 0] + h_shift, 0, 179)
        adjusted_hsv[:, :, 1] = np.clip(adjusted_hsv[:, :, 1] + s_shift, 0, 255)
        adjusted_hsv[:, :, 2] = np.clip(adjusted_hsv[:, :, 2] + v_shift, 0, 255)
        
        # 5. Apply directional lighting
        # Create a gradient mask based on the light direction
        y_coords, x_coords = np.mgrid[0:prod_height, 0:prod_width]
        
        # Normalize coordinates to center
        x_centered = x_coords - prod_width // 2
        y_centered = y_coords - prod_height // 2
        
        # Calculate the dot product with the light direction
        light_angle = light_direction + np.pi  # Reverse the angle to get illumination direction
        direction_x = np.cos(light_angle)
        direction_y = np.sin(light_angle)
        
        # Dot product to get lighting intensity
        light_intensity = x_centered * direction_x + y_centered * direction_y
        
        # Normalize to 0-1 range
        light_intensity = light_intensity - np.min(light_intensity)
        if np.max(light_intensity) > 0:
            light_intensity = light_intensity / np.max(light_intensity)
        
        # Scale by the light strength and background variance
        light_effect = light_intensity * light_strength * (bg_std_v / 128) * 20
        
        # Apply the directional lighting effect
        adjusted_hsv[:, :, 2] = np.clip(adjusted_hsv[:, :, 2] + light_effect, 0, 255)
        
        # Convert back to RGB
        adjusted_rgb = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2RGB)
        
        # Create the final adjusted product
        lighting_adjusted_product = product_np.copy()
        lighting_adjusted_product[:, :, :3] = adjusted_rgb
        
        return lighting_adjusted_product
    
    def blend_images(self, product, background, mask, position):
        """Use inpainting to seamlessly blend the product into the scene"""
        # Convert images to numpy arrays if they're PIL Images
        if isinstance(product, Image.Image):
            product_np = np.array(product)
        else:
            product_np = product.copy()
        
        if isinstance(background, Image.Image):
            bg_np = np.array(background)
            bg_pil = background
        else:
            bg_np = background.copy()
            bg_pil = Image.fromarray(bg_np)
        
        # Extract dimensions
        prod_height, prod_width = product_np.shape[:2]
        bg_height, bg_width = bg_np.shape[:2]
        x_pos, y_pos = position
        
        # Create a copy of the background
        composite = bg_np.copy()
        
        # Extract alpha channel from product if available
        if product_np.shape[2] == 4:  # RGBA
            alpha = product_np[:, :, 3] / 255.0
            product_rgb = product_np[:, :, :3]
        else:
            # Use provided mask if product doesn't have alpha
            alpha = mask.astype(float)
            product_rgb = product_np
        
        # Ensure alpha is 2D
        if len(alpha.shape) > 2:
            alpha = alpha[:, :, 0]
        
        # Calculate the valid region for placement (handles edge cases)
        valid_h = min(prod_height, bg_height - y_pos)
        valid_w = min(prod_width, bg_width - x_pos)
        
        if valid_h <= 0 or valid_w <= 0:
            return Image.fromarray(bg_np)  # Return original background if placement is invalid
        
        # Create the initial composite using alpha blending
        for c in range(3):  # RGB channels
            composite[y_pos:y_pos+valid_h, x_pos:x_pos+valid_w, c] = (
                product_rgb[:valid_h, :valid_w, c] * alpha[:valid_h, :valid_w] + 
                composite[y_pos:y_pos+valid_h, x_pos:x_pos+valid_w, c] * (1 - alpha[:valid_h, :valid_w])
            )
        
        # Convert to PIL Image for further processing
        composite_pil = Image.fromarray(composite)
        
        # Create a mask for the area around the product edges
        # We'll use this for the inpainting process
        edge_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        
        # Dilate the product mask to get the edges
        if np.any(alpha):  # Check if alpha has any non-zero values
            # Create a binary mask from alpha
            binary_mask = (alpha > 0.1).astype(np.uint8)
            
            # Dilate the mask to get the area around the edges
            kernel = np.ones((15, 15), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            # Subtract the original mask to get just the edges
            edge_binary = dilated_mask - binary_mask
            
            # Place the edge mask in the correct position
            edge_mask[y_pos:y_pos+valid_h, x_pos:x_pos+valid_w] = edge_binary[:valid_h, :valid_w] * 255
        
        # Convert mask to PIL
        edge_mask_pil = Image.fromarray(edge_mask)
        
        # Prepare the prompt for stable diffusion inpainting
        # Analyze the background to generate an appropriate description
        bg_description = self.generate_background_description(bg_pil)
        product_type = self.identify_product_type(product_np)
        
        # Create a context-aware prompt for the inpainting
        prompt = f"A realistic photograph of {product_type} in {bg_description}, seamless integration, natural lighting, photorealistic"
        
        # Use stable diffusion for inpainting if the edge mask is not empty
        if np.any(edge_mask):
            try:
                # Scale down for faster processing if needed
                max_size = 768  # Max size for inpainting
                if max(bg_width, bg_height) > max_size:
                    scale_factor = max_size / max(bg_width, bg_height)
                    scaled_width = int(bg_width * scale_factor)
                    scaled_height = int(bg_height * scale_factor)
                    
                    composite_pil_scaled = composite_pil.resize((scaled_width, scaled_height), Image.LANCZOS)
                    edge_mask_pil_scaled = edge_mask_pil.resize((scaled_width, scaled_height), Image.LANCZOS)
                    
                    # Run inpainting on scaled version
                    inpainted_image = self.inpaint_pipeline(
                        prompt=prompt,
                        image=composite_pil_scaled,
                        mask_image=edge_mask_pil_scaled,
                        guidance_scale=7.5,
                        num_inference_steps=25,  # Lower for speed
                    ).images[0]
                    
                    # Scale back to original size
                    inpainted_image = inpainted_image.resize((bg_width, bg_height), Image.LANCZOS)
                else:
                    # Run inpainting at original size
                    inpainted_image = self.inpaint_pipeline(
                        prompt=prompt,
                        image=composite_pil,
                        mask_image=edge_mask_pil,
                        guidance_scale=7.5,
                        num_inference_steps=30,
                    ).images[0]
                
                # Create a final composite using the inpainted image
                # This keeps the original product intact and only blends the edges
                inpainted_np = np.array(inpainted_image)
                
                # Create a smooth transition mask for blending
                smooth_mask = cv2.GaussianBlur(alpha[:valid_h, :valid_w], (15, 15), 0)
                
                # Final composite
                final_composite_np = composite.copy()
                
                # Keep the original product where alpha is high, and use inpainted result where alpha is low
                # This ensures the product details remain intact
                inpaint_region = final_composite_np[y_pos:y_pos+valid_h, x_pos:x_pos+valid_w]
                inpaint_src = inpainted_np[y_pos:y_pos+valid_h, x_pos:x_pos+valid_w]
                
                # Only copy valid regions
                valid_inpaint_h = min(inpaint_region.shape[0], inpaint_src.shape[0])
                valid_inpaint_w = min(inpaint_region.shape[1], inpaint_src.shape[1])
                
                if valid_inpaint_h > 0 and valid_inpaint_w > 0:
                    mask_region = smooth_mask[:valid_inpaint_h, :valid_inpaint_w]
                    inpaint_region[:valid_inpaint_h, :valid_inpaint_w] = (
                        product_rgb[:valid_inpaint_h, :valid_inpaint_w] * mask_region[:, :, np.newaxis] +
                        inpaint_src[:valid_inpaint_h, :valid_inpaint_w] * (1 - mask_region[:, :, np.newaxis])
                    )
                
                final_composite = Image.fromarray(final_composite_np)
                
            except Exception as e:
                print(f"Inpainting error: {e}. Falling back to standard alpha blending.")
                final_composite = composite_pil
        else:
            # If no edge mask (e.g., very small product), just use the alpha-blended composite
            final_composite = composite_pil
        
        return final_composite

    def generate_background_description(self, background_image):
        """Generate a description of the background scene using CLIP"""
        # Preprocess the image
        inputs = self.clip_processor(
            text=["living room", "bedroom", "kitchen", "office", "outdoor scene"],
            images=background_image,
            return_tensors="pt",
            padding=True
        )
        
        # Get model outputs
        outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get top scene type
        top_scene_idx = probs.argmax().item()
        scene_types = ["living room", "bedroom", "kitchen", "office", "outdoor scene"]
        
        return scene_types[top_scene_idx]

    def identify_product_type(self, product_image):
        """Identify the type of product using CLIP"""
        # Convert to PIL for CLIP processing
        if not isinstance(product_image, Image.Image):
            if product_image.shape[2] == 4:  # RGBA
                product_pil = Image.fromarray(product_image[:, :, :3])
            else:
                product_pil = Image.fromarray(product_image)
        else:
            product_pil = product_image
        
        # Common product types for e-commerce
        product_types = [
            "a vase", "a lamp", "decorative cushions", "a chair", "a desk", 
            "a bookshelf", "wall art", "a clock", "a plant pot", "a decorative figurine",
            "a candle holder", "a coffee table", "a side table", "home decor item"
        ]
        
        # Preprocess the image
        inputs = self.clip_processor(
            text=product_types,
            images=product_pil,
            return_tensors="pt",
            padding=True
        )
        
        # Get model outputs
        outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get top product type
        top_product_idx = probs.argmax().item()
        
        return product_types[top_product_idx]
    def select_best_placement(self, placement_areas, product_no_bg):
        """Select the best placement area based on product size and scene context"""
        # If no placement areas found, return a default area
        if not placement_areas or len(placement_areas) == 0:
            return {"x": 0, "y": 0, "width": 100, "height": 100, "score": 0.5}
        
        # Convert product to numpy array if it's a PIL Image
        if isinstance(product_no_bg, Image.Image):
            product_array = np.array(product_no_bg)
        else:
            product_array = product_no_bg
        
        # Get product dimensions
        product_height, product_width = product_array.shape[:2]
        
        # Find the best placement area based on size compatibility and score
        best_area = None
        best_score = -1
        
        for area in placement_areas:
            # Check if product fits in this area with reasonable scaling
            area_ratio = min(area["width"] / product_width, area["height"] / product_height)
            
            # Skip areas that would require extreme scaling
            if area_ratio < 0.2 or area_ratio > 5:
                continue
            
            # Calculate a combined score based on area score and fit
            fit_score = 1 - abs(1 - area_ratio)  # 1 is perfect fit, 0 is poor fit
            combined_score = area["score"] * 0.7 + fit_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_area = area
        
        # If no suitable area found, return the highest scoring area
        if best_area is None and len(placement_areas) > 0:
            best_area = max(placement_areas, key=lambda x: x["score"])
        
        return best_area
    
    def process_image_pair(self, product_path, background_path):
        """Process a single product and background pair"""
        try:
            # Load images
            product_img = Image.open(product_path)
            background_img = Image.open(background_path)
            
            # Debugging image sizes
            print(f"Product Image size: {product_img.size}")  # (width, height)
            print(f"Background Image size: {background_img.size}")  # (width, height)

            if not product_img or not background_img:
                raise ValueError("One of the images failed to load correctly.")

            # Remove background from product
            product_no_bg = self.remove_background(product_img)
            if product_no_bg is None:
                raise ValueError("Background removal failed. 'product_no_bg' is None.")
            
            # Debugging the result of background removal
            print(f"Product after background removal size: {product_no_bg.size}")

            # Analyze scene for placement
            placement_areas = self.analyze_scene(background_img)
            if not placement_areas:
                raise ValueError("No valid placement areas found.")

            # Debugging placement areas
            print(f"Placement areas: {placement_areas}")

            # Choose best placement
            best_area = self.select_best_placement(placement_areas, product_no_bg)
            if best_area is None:
                raise ValueError("Best placement area not found.")

            # Transform product for placement
            transformed_product, mask, position = self.optimize_placement(
                product_no_bg, background_img, best_area)
            if transformed_product is None or mask is None:
                raise ValueError("Product transformation failed.")
            
            # Debugging transformed product
            print(f"Transformed product size: {transformed_product.size}")

            # Adjust lighting
            lit_product = self.adjust_lighting(transformed_product, background_img, position)
            if lit_product is None:
                raise ValueError("Lighting adjustment failed.")
            
            # Debugging lit product
            print(f"Lit product size: {lit_product.size}")

            # Blend images
            final_result = self.blend_images(lit_product, background_img, mask, position)
            if final_result is None:
                raise ValueError("Blending failed.")
            
            # Debugging final result
            print(f"Final result size: {final_result.size}")

            return final_result

        except Exception as e:
            print(f"Error during processing: {e}")
            raise

    def batch_process(self, product_dir, background_dir, output_dir):
        """Process multiple product-background pairs"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Map products to backgrounds
        product_files = [f for f in os.listdir(product_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        background_files = [f for f in os.listdir(background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each pair
        for i, product_file in enumerate(product_files):
            # Select background (cyclically or based on product type)
            bg_idx = i % len(background_files)
            background_file = background_files[bg_idx]
            
            # Process the pair
            result = self.process_image_pair(
                os.path.join(product_dir, product_file),
                os.path.join(background_dir, background_file)
            )
            
            # Save result
            output_path = os.path.join(output_dir, f"composite_{os.path.splitext(product_file)[0]}.png")
            result.save(output_path)
            
            print(f"Processed {product_file} with {background_file} -> {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Product Image Integration System")
    parser.add_argument("--product_dir", required=True, help="Directory containing product images")
    parser.add_argument("--background_dir", required=True, help="Directory containing background images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output composites")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run models on")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    integrator = ProductImageIntegrator(device=args.device)
    integrator.batch_process(args.product_dir, args.background_dir, args.output_dir)