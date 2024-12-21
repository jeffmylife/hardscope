import os
import json
from pathlib import Path
from typing import Dict, Union
import base64
from io import BytesIO
import sys

from PIL import Image, ImageDraw
from litellm import completion
from dotenv import load_dotenv
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)
console = Console()

load_dotenv()

def add_grid(image: Image.Image, grid_size: int = 10) -> Image.Image:
    """Add a grid overlay to the image.
    
    Args:
        image: PIL Image to add grid to
        grid_size: Number of divisions in each dimension
    
    Returns:
        New image with grid overlay
    """
    # Create a copy of the image with a white border
    bordered_img = Image.new('RGBA', (image.width + 60, image.height + 60), (255, 255, 255, 255))
    bordered_img.paste(image, (30, 30))
    img = bordered_img
    draw = ImageDraw.Draw(img, 'RGBA')
    
    width, height = image.width, image.height
    cell_width = width / grid_size
    cell_height = height / grid_size
    
    # Starting points for the grid (after border)
    start_x, start_y = 30, 30
    
    # Draw outer border (thick red line)
    draw.rectangle(
        [(start_x-1, start_y-1), (start_x+width+1, start_y+height+1)],
        outline=(255, 0, 0, 255),
        width=3
    )
    
    # Draw vertical lines
    for i in range(grid_size + 1):
        x = start_x + i * cell_width
        draw.line(
            [(x, start_y), (x, start_y+height)],
            fill=(255, 0, 0, 200),
            width=2
        )
    
    # Draw horizontal lines
    for i in range(grid_size + 1):
        y = start_y + i * cell_height
        draw.line(
            [(start_x, y), (start_x+width, y)],
            fill=(255, 0, 0, 200),
            width=2
        )
    
    # Add crosshairs at grid intersections
    for row in range(grid_size + 1):
        for col in range(grid_size + 1):
            x = start_x + col * cell_width
            y = start_y + row * cell_height
            size = 4
            # Black crosshair with white outline for visibility
            for offset in [-1, 1]:
                draw.line([(x-size, y+offset), (x+size, y+offset)], fill=(255, 255, 255, 255), width=2)
                draw.line([(x+offset, y-size), (x+offset, y+size)], fill=(255, 255, 255, 255), width=2)
            draw.line([(x-size, y), (x+size, y)], fill=(0, 0, 0, 255), width=1)
            draw.line([(x, y-size), (x, y+size)], fill=(0, 0, 0, 255), width=1)
    
    # Add coordinate markers with larger font
    try:
        font_size = 24  # Fixed larger font size
        cell_font_size = int(min(cell_width, cell_height) / 2)  # Increased from /3 to /2
        from PIL import ImageFont
        font = ImageFont.truetype("Arial", font_size)
        cell_font = ImageFont.truetype("Arial Bold", cell_font_size)
    except:
        font = None
        cell_font = None
    
    # Add column markers (A, B, C, ...)
    for i in range(grid_size):
        col_label = chr(65 + i)  # A=65 in ASCII
        x = start_x + i * cell_width + cell_width/2
        # Draw text with better contrast
        draw.text((x, 15), col_label, fill=(255, 0, 0, 255), font=font, anchor="mm", stroke_width=2, stroke_fill=(255, 255, 255, 255))
    
    # Add row markers (1, 2, 3, ...)
    for i in range(grid_size):
        row_label = str(i + 1)
        y = start_y + i * cell_height + cell_height/2
        # Draw text with better contrast
        draw.text((15, y), row_label, fill=(255, 0, 0, 255), font=font, anchor="mm", stroke_width=2, stroke_fill=(255, 255, 255, 255))
    
    # Add cell labels (A1, A2, B1, B2, etc.)
    for row in range(grid_size):
        for col in range(grid_size):
            cell_label = f"{chr(65 + col)}{row + 1}"
            x = start_x + col * cell_width + cell_width/2
            y = start_y + row * cell_height + cell_height/2
            
            # Add subtle cell highlighting
            highlight_cell_boundaries(draw, cell_label, grid_size, cell_width, cell_height, start_x, start_y)
            
            # Draw white background with padding
            label_bbox = draw.textbbox((x, y), cell_label, font=cell_font, anchor="mm")
            padding = cell_font_size // 3
            bg_bbox = (
                label_bbox[0] - padding,
                label_bbox[1] - padding,
                label_bbox[2] + padding,
                label_bbox[3] + padding
            )
            draw.rectangle(bg_bbox, fill=(255, 255, 255, 240))
            
            # Draw text with thick black outline for maximum contrast
            for offset_x, offset_y in [(-3,-3), (-3,3), (3,-3), (3,3)]:  # Increased outline thickness
                draw.text((x+offset_x, y+offset_y), cell_label, fill=(0, 0, 0, 255), font=cell_font, anchor="mm")
            
            # Draw the main text in red
            draw.text((x, y), cell_label, fill=(255, 0, 0, 255), font=cell_font, anchor="mm")
    
    return img

def save_staging_image(img: Image.Image, image_path: Path) -> str:
    """Save the image that will be sent to the model."""
    os.makedirs("staging-images", exist_ok=True)
    output_path = f"staging-images/{Path(image_path).stem}_with_grid{Path(image_path).suffix}"
    img.save(output_path)
    console.print(f"[blue]Saved staging image to: {output_path}[/blue]")
    return output_path

def encode_image(image_path: Union[str, Path], should_add_grid: bool = True, grid_size: int = 10) -> str:
    """Convert an image to base64 string, optionally adding a grid overlay."""
    with Image.open(image_path) as img:
        if should_add_grid:
            gridded_img = add_grid(img, grid_size)
            save_staging_image(gridded_img, image_path)
            img = gridded_img
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

def draw_prediction(image_path: Union[str, Path], result: Dict, target: str) -> None:
    """Draw bounding box on image and save the result."""
    if not result["found"]:
        return
    
    # Open image and create draw object
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    # The coordinates are already normalized (0-1), so we need to denormalize them
    coords = result["coordinates"]
    x1 = int(coords["x1"] * width)
    y1 = int(coords["y1"] * height)
    x2 = int(coords["x2"] * width)
    y2 = int(coords["y2"] * height)
    
    # Draw rectangle
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    
    # Add label with confidence and grid cells
    label = f"{target} ({result['confidence']:.2f})"
    if "grid_cells" in result:
        label += f" {result['grid_cells']}"
    
    font_size = max(12, min(width, height) // 50)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("Arial", font_size)
    except:
        font = None
    
    # Draw label background
    label_bbox = draw.textbbox((x1, y1-font_size-4), label, font=font)
    draw.rectangle(label_bbox, fill="red")
    draw.text((x1, y1-font_size-4), label, fill="white", font=font)
    
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Save the image
    output_path = f"predictions/{Path(image_path).stem}_pred{Path(image_path).suffix}"
    img.save(output_path)
    console.print(f"[green]Prediction saved to: {output_path}[/green]")

def detect_object(
    image_path: Union[str, Path],
    target: str,
    model: str = "gpt-4o",
    grid_size: int = 15
) -> Dict:
    """
    Detect specified target in an image.
    
    Args:
        image_path: Path to the image file
        target: Description of what to find (e.g., "cat", "red car", "person wearing hat")
        model: The model to use for detection
        grid_size: Size of the grid overlay (e.g., 10 for 10x10 grid)
    
    Returns:
        Dictionary containing detection results with bounding box coordinates
    """
    console.print(f"[blue]Processing image: {image_path}[/blue]")
    console.print(f"[blue]Looking for: {target}[/blue]")
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
    console.print(f"[blue]Image dimensions: {width}x{height}[/blue]")
    
    # Calculate grid cell dimensions
    cell_width = width / grid_size
    cell_height = height / grid_size
    console.print(f"[blue]Grid cell size: {cell_width:.1f}x{cell_height:.1f} pixels[/blue]")
    
    # Calculate last column label (A-J for grid_size=10)
    last_col = chr(64 + grid_size)  # A=65 in ASCII
    
    # Get cell boundaries
    cell_boundaries = get_cell_boundaries(grid_size, width, height)
    
    # Format cell information
    grid_info = []
    # Header row showing column coordinates
    header = "     " + "  ".join([f"{chr(65+i):^15}" for i in range(grid_size)])
    grid_info.append(header)
    
    # Add cell information in a grid format
    for row in range(grid_size):
        row_cells = []
        row_label = f"{row+1:2d} |"
        for col in range(grid_size):
            cell_id = f"{chr(65 + col)}{row + 1}"
            bounds = cell_boundaries[cell_id]
            cell_info = f"({bounds['x1']},{bounds['y1']})-({bounds['x2']},{bounds['y2']})"
            row_cells.append(f"{cell_info:^15}")
        grid_info.append(f"{row_label} " + "  ".join(row_cells))
    
    # Add example section
    grid_info.append("\nExample Grid References:")
    grid_info.append("1. Single cell: 'A1' refers to the top-left cell")
    grid_info.append("2. Multiple cells: 'B3-C4' covers a 2x2 area from B3 to C4")
    grid_info.append("3. For states/large objects: Use the smallest rectangle that contains the entire object")
    grid_info.append("   Example: Louisiana typically spans I11-J13 (a 2x3 grid of cells)")
    
    # Join all grid information
    grid_cells_section = "\n".join(grid_info)
    
    # Read and format prompt
    with open("prompt.txt", "r") as f:
        prompt_template = f.read()
        prompt = prompt_template.format(
            width=width,
            height=height,
            grid_size=grid_size,
            cell_width=cell_width,
            cell_height=cell_height,
            last_column=last_col,
            grid_cells=grid_cells_section
        )
    
    # Save the generated prompt for inspection
    with open("generated_prompt.txt", "w") as f:
        f.write(prompt)
    
    # Encode the image with grid
    base64_image = encode_image(image_path, should_add_grid=True, grid_size=grid_size)
    console.print("[blue]Image encoded successfully with grid overlay[/blue]")
    
    # Prepare the messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}\n\nFind this element in the image: {target}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    console.print(f"[blue]Requesting prediction from model: {model}[/blue]")
    response = completion(
        model=model,
        messages=messages,
        max_tokens=1000,
        response_format={"type": "json_object"}
    )
    
    # Get the raw response and clean it
    raw_content = response.choices[0].message.content.strip()
    console.print("\n[yellow]Raw model response:[/yellow]")
    console.print(raw_content)
    
    # Clean the response - remove any markdown, newlines, or extra spaces
    content = raw_content
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip().replace('\n', '').replace('  ', ' ')
    
    result = json.loads(content)
    
    # Validate the result structure
    required_keys = {"found", "coordinates", "confidence"}
    if not all(k in result for k in required_keys):
        missing = required_keys - set(result.keys())
        raise ValueError(f"Missing required keys in response: {missing}")
    
    # Validate coordinates if found
    if result["found"]:
        coords = result["coordinates"]
        if not isinstance(coords, dict):
            raise ValueError(f"Coordinates must be a dict, got {type(coords)}")
        
        required_coords = {"x1", "y1", "x2", "y2"}
        if not all(k in coords for k in required_coords):
            missing = required_coords - set(coords.keys())
            raise ValueError(f"Missing coordinate keys: {missing}")
        
        # Validate coordinate values
        if not all(isinstance(coords[k], (int, float)) for k in required_coords):
            raise ValueError("Coordinate values must be numbers")
        
        if not (0 <= coords["x1"] < coords["x2"] <= width):
            raise ValueError(f"Invalid x coordinates: {coords['x1']}, {coords['x2']}")
        
        if not (0 <= coords["y1"] < coords["y2"] <= height):
            raise ValueError(f"Invalid y coordinates: {coords['y1']}, {coords['y2']}")
        
        # Normalize the coordinates
        result["coordinates"] = {
            "x1": coords["x1"] / width,
            "y1": coords["y1"] / height,
            "x2": coords["x2"] / width,
            "y2": coords["y2"] / height
        }
        console.print(f"[green]Found {target} at coordinates: {coords}[/green]")
    else:
        if result["coordinates"] is not None:
            raise ValueError("If found is false, coordinates must be null")
        console.print(f"[yellow]{target} not found in image[/yellow]")
    
    return result

def get_cell_boundaries(grid_size: int, width: int, height: int) -> Dict:
    """Generate exact pixel boundaries for each cell.
    
    Args:
        grid_size: Size of the grid (NxN)
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Dictionary mapping cell IDs to their pixel boundaries
    """
    cell_width = width / grid_size
    cell_height = height / grid_size
    
    boundaries = {}
    for row in range(grid_size):
        for col in range(grid_size):
            cell_id = f"{chr(65 + col)}{row + 1}"
            boundaries[cell_id] = {
                "x1": int(col * cell_width),
                "y1": int(row * cell_height),
                "x2": int((col + 1) * cell_width),
                "y2": int((row + 1) * cell_height)
            }
    return boundaries

def highlight_cell_boundaries(draw: ImageDraw, cell_id: str, grid_size: int, 
                            cell_width: float, cell_height: float, 
                            start_x: int, start_y: int) -> None:
    """Add subtle highlighting to show cell boundaries.
    
    Args:
        draw: PIL ImageDraw object
        cell_id: Cell identifier (e.g., 'A1')
        grid_size: Size of the grid (NxN)
        cell_width: Width of each cell
        cell_height: Height of each cell
        start_x: Starting x coordinate for grid
        start_y: Starting y coordinate for grid
    """
    col = ord(cell_id[0]) - 65
    row = int(cell_id[1:]) - 1
    
    x1 = start_x + col * cell_width
    y1 = start_y + row * cell_height
    x2 = x1 + cell_width
    y2 = y1 + cell_height
    
    # Draw subtle fill
    draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 0, 0, 30))
    
    # Draw diagonal lines for better visibility
    line_spacing = 20
    for offset in range(0, int(cell_width + cell_height), line_spacing):
        draw.line([(x1 + offset, y1), (x1, y1 + offset)], fill=(255, 0, 0, 40), width=1)
        draw.line([(x2 - offset, y1), (x2, y1 + offset)], fill=(255, 0, 0, 40), width=1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect objects in images using LLM")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("target", type=str, help="What to find in the image")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                       help="Model to use for detection")
    parser.add_argument("--grid-size", type=int, default=15,
                       help="Size of the grid overlay (NxN)")
    
    args = parser.parse_args()
    
    result = detect_object(args.image_path, args.target, model=args.model, grid_size=args.grid_size)
    if result["found"]:
        console.print(f"[green]Found {args.target} with confidence: {result['confidence']:.2f}[/green]")
        draw_prediction(args.image_path, result, args.target)
    else:
        console.print(f"[yellow]Could not find {args.target} in the image[/yellow]")