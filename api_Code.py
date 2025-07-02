import os
import shutil
import argparse
from gradio_client import Client, file


def generate_3d_mesh_from_image(
    input_image_path: str,
    output_dir: str = "instantmesh_output",
    do_remove_background: bool = True,
    sample_steps: float = 75,
    sample_seed: float = 42
) -> tuple[str, str] | None:
    """
    Generates a 3D mesh (OBJ and GLB formats) from an input image using the
    TencentARC/InstantMesh Gradio API.

    Args:
        input_image_path (str): Path to the local input image file.
        output_dir (str): Directory to save the generated 3D models.
                          Defaults to "instantmesh_output".
        do_remove_background (bool): Whether to remove the background during preprocessing.
                                     Defaults to True.
        sample_steps (float): Number of sample steps for multi-view generation.
                              Defaults to 75.
        sample_seed (float): Seed value for multi-view generation. Defaults to 42.

    Returns:
        tuple[str, str] | None: A tuple containing the paths to the saved OBJ and GLB
                                files, or None if an error occurred.
    """
    # 1. Validate input image path
    if not os.path.exists(input_image_path):
        print(f"Error: Input image file not found at '{input_image_path}'.")
        return None

    # 2. Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}': {e}")
        return None

    client = None
    try:
        print(f"Connecting to InstantMesh Gradio API at 'TencentARC/InstantMesh'...")
        # Initialize the Gradio client
        client = Client("TencentARC/InstantMesh")
        print("Connection successful.")

        # 3. Step 1: Preprocess the input image
        print(
            f"Calling '/preprocess' with input image '{input_image_path}' (remove_background: {do_remove_background})...")
        processed_image_gradio_path = client.predict(
            input_image=file(input_image_path),
            do_remove_background=do_remove_background,
            api_name="/preprocess"
        )
        print(
            f"'/preprocess' completed. Result: {processed_image_gradio_path}")

        # Save the preprocessed image
        preprocessed_filename = os.path.join(output_dir, "preprocessed_image.png")
        print(f"Saving preprocessed image to: {preprocessed_filename}")
        shutil.copy(processed_image_gradio_path, preprocessed_filename)

        # 4. Step 2: Generate multi-views from the processed image
        # The `gradio_client` handles passing the internal file reference from the previous step.
        print(
            f"Calling '/generate_mvs' with processed image (sample_steps: {sample_steps}, sample_seed: {sample_seed})...")
        generated_mvs_gradio_path = client.predict(
            # This passes the temporary file path from the server
            input_image=file(processed_image_gradio_path),
            sample_steps=sample_steps,
            sample_seed=sample_seed,
            api_name="/generate_mvs"
        )
        print(
            f"'/generate_mvs' completed. Result: {generated_mvs_gradio_path}")

        # Save the multi-view generated image
        multiview_filename = os.path.join(output_dir, "multiview_generation.png")
        print(f"Saving multi-view generation to: {multiview_filename}")
        shutil.copy(generated_mvs_gradio_path, multiview_filename)

        # 5. Step 3: Generate 3D model (OBJ and GLB)
        # This step operates on the internal state after multi-view generation.
        print("Calling '/make3d' to generate OBJ and GLB models...")
        obj_model_temp_path, glb_model_temp_path = client.predict(
            api_name="/make3d"
        )
        print(
            f"'/make3d' completed. Temporary OBJ: {obj_model_temp_path}, Temporary GLB: {glb_model_temp_path}")

        # 6. Save the generated models to the specified output directory
        obj_filename = os.path.join(
            output_dir, os.path.basename(obj_model_temp_path))
        glb_filename = os.path.join(
            output_dir, os.path.basename(glb_model_temp_path))

        print(f"Saving generated OBJ to: {obj_filename}")
        shutil.copy(obj_model_temp_path, obj_filename)
        print(f"Saving generated GLB to: {glb_filename}")
        shutil.copy(glb_model_temp_path, glb_filename)

        print(f"\nSuccessfully generated and saved 3D models:")
        print(f"  Preprocessed image: {preprocessed_filename}")
        print(f"  Multi-view generation: {multiview_filename}")
        print(f"  OBJ format: {obj_filename}")
        print(f"  GLB format: {glb_filename}")

        return obj_filename, glb_filename

    except Exception as e:
        print(
            f"\nAn error occurred during API interaction or file operations: {e}")
        # For more granular error handling, you could catch specific exceptions like
        # gradio_client.exceptions.AppError for API-specific errors,
        # requests.exceptions.ConnectionError for network issues, etc.
        return None
    finally:
        # Ensure the client connection is properly closed.
        if client:
            client.close()


# --- Example Usage ---
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from an input image using InstantMesh API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_code.py table_colour.jpg
  python api_code.py table_colour.jpg --output my_outputs
  python api_code.py table_colour.jpg --output my_outputs --steps 100 --seed 456
        """
    )

    parser.add_argument(
        "input_image",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--output", "-o",
        default="my_instantmesh_outputs",
        help="Output directory for generated 3D models (default: my_instantmesh_outputs)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=75,
        help="Number of sample steps for multi-view generation (default: 75)"
    )
    parser.add_argument(
        "--seed", "-r",
        type=int,
        default=42,
        help="Seed value for multi-view generation (default: 42)"
    )
    parser.add_argument(
        "--keep-background", "-k",
        action="store_true",
        help="Keep the background (default: remove background)"
    )

    args = parser.parse_args()

    # Use command-line arguments
    local_input_image = args.input_image
    output_directory_for_meshes = args.output
    sample_steps = args.steps
    sample_seed = args.seed
    remove_background = not args.keep_background

    if os.path.exists(local_input_image):
        print(f"Using local image: '{local_input_image}'")
        print(f"Output directory: '{output_directory_for_meshes}'")
        print(f"Sample steps: {sample_steps}")
        print(f"Sample seed: {sample_seed}")
        print(f"Remove background: {remove_background}")

        print(
            f"\n--- Starting 3D mesh generation for '{local_input_image}' ---")
        generated_files = generate_3d_mesh_from_image(
            input_image_path=local_input_image,
            output_dir=output_directory_for_meshes,
            do_remove_background=remove_background,
            sample_steps=sample_steps,
            sample_seed=sample_seed
        )

        if generated_files:
            print("\nInstantMesh process completed successfully.")
        else:
            print("\nInstantMesh process failed.")
    else:
        print(f"\nError: Input image file '{local_input_image}' not found.")
        print("Please check the file path and try again.")

    print("\nScript execution finished.")
