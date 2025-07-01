import os
import shutil
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
        print(f"Calling '/preprocess' with input image '{input_image_path}' (remove_background: {do_remove_background})...")
        processed_image_gradio_path = client.predict(
            input_image=file(input_image_path),
            do_remove_background=do_remove_background,
            api_name="/preprocess"
        )
        print(f"'/preprocess' completed. Result: {processed_image_gradio_path}")

        # 4. Step 2: Generate multi-views from the processed image
        # The `gradio_client` handles passing the internal file reference from the previous step.
        print(f"Calling '/generate_mvs' with processed image (sample_steps: {sample_steps}, sample_seed: {sample_seed})...")
        generated_mvs_gradio_path = client.predict(
            input_image=file(processed_image_gradio_path), # This passes the temporary file path from the server
            sample_steps=sample_steps,
            sample_seed=sample_seed,
            api_name="/generate_mvs"
        )
        print(f"'/generate_mvs' completed. Result: {generated_mvs_gradio_path}")

        # 5. Step 3: Generate 3D model (OBJ and GLB)
        # This step operates on the internal state after multi-view generation.
        print("Calling '/make3d' to generate OBJ and GLB models...")
        obj_model_temp_path, glb_model_temp_path = client.predict(
            api_name="/make3d"
        )
        print(f"'/make3d' completed. Temporary OBJ: {obj_model_temp_path}, Temporary GLB: {glb_model_temp_path}")

        # 6. Save the generated models to the specified output directory
        obj_filename = os.path.join(output_dir, os.path.basename(obj_model_temp_path))
        glb_filename = os.path.join(output_dir, os.path.basename(glb_model_temp_path))

        print(f"Saving generated OBJ to: {obj_filename}")
        shutil.copy(obj_model_temp_path, obj_filename)
        print(f"Saving generated GLB to: {glb_filename}")
        shutil.copy(glb_model_temp_path, glb_filename)

        print(f"\nSuccessfully generated and saved 3D models:")
        print(f"  OBJ format: {obj_filename}")
        print(f"  GLB format: {glb_filename}")

        return obj_filename, glb_filename

    except Exception as e:
        print(f"\nAn error occurred during API interaction or file operations: {e}")
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
    # Use the local image file in the directory
    local_input_image = 'Mobile-Phone-Drawing-08.jpg'

    if os.path.exists(local_input_image):
        print(f"Using local image: '{local_input_image}'")
        output_directory_for_meshes = "my_instantmesh_outputs"
        print(f"\n--- Starting 3D mesh generation for '{local_input_image}' ---")
        generated_files = generate_3d_mesh_from_image(
            input_image_path=local_input_image,
            output_dir=output_directory_for_meshes,
            do_remove_background=True,
            sample_steps=75,
            sample_seed=124 # 42
        )

        if generated_files:
            print("\nInstantMesh process completed successfully.")
        else:
            print("\nInstantMesh process failed.")
    else:
        print("\nSkipping 3D mesh generation due to missing input image.")

    print("\nScript execution finished.")