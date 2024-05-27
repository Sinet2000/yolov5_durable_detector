import subprocess
import os
import shutil
import time

from models.image_prediction_result import ImagePredictionResult, DetectorType

def rename_file_with_prefix(file_path, prefix) -> tuple:
    try:
        if os.path.exists(file_path):
            # Split the file path into directory and filename
            directory, filename = os.path.split(file_path)
            # Append the prefix to the filename
            new_filename = f"{prefix}{filename}"
            # Construct the new file path
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"File '{file_path}' renamed to '{new_file_path}'.")

            return new_filename, new_file_path
        else:
            print(f"File '{file_path}' does not exist.")
            return None
    except OSError as e:
        print(f"Error renaming file: {e}")
        return None

def run_yolov_detector(
        file_name,
        source_path,
        script_dir,
        csv_path,
        result_dir_name,
        prediction_result_dir
        ) -> ImagePredictionResult:
    
    if os.path.exists(prediction_result_dir):
        # Remove the directory and its contents
        shutil.rmtree(prediction_result_dir)

    # Save the current working directory
    original_dir = os.getcwd()
    detector_type = DetectorType.YoloV5
    # Define the command to run
    try:
        # Change the working directory to script_dir
        os.chdir(script_dir)

        # Define the command to run
        command = [
            "python", "detect.py",
            "--source", source_path,
            "--weights", "runs/train/yolov-202405-last.pt",
            "--conf", "0.25",
            "--project", "runs/detect",
            "--name", result_dir_name,
            "--save-csv"
        ]

        # Run the command and capture output and errors
        start_time = time.time()
        process = subprocess.run(' '.join(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        end_time = time.time()
        time_taken = end_time - start_time

        errors = process.stderr.decode()

        # Log the output and errors
        output = process.stdout.decode()
        errors = process.stderr.decode()
        if output:
            print("Output:", output)
        if errors:
            print("Errors:", errors)

        try:
            with open(csv_path, "r") as file:
                predictions_csv_content = file.read()
        except FileNotFoundError:
            predictions_csv_content = "Predictions CSV file not found"

        # Print the output and errors after the result
        if output:
            print("Output:", output)
        if errors:
            print("Errors:", errors)

        # Serialize the output to JSON
        result_img_name, result_img_path = rename_file_with_prefix(os.path.join(prediction_result_dir, file_name), 'yolov5_')
        first_row = predictions_csv_content.split('\n')[0]
        # Splitting the first row into components 
        image_name, classification, prediction = first_row.split(',')

        return ImagePredictionResult(
            image_name=image_name,
            detector_type=detector_type,
            classification = classification,
            result_img_name=result_img_name,
            result_img_path=result_img_path,
            prediction=float(prediction),
            time_taken=time_taken)
    
    except Exception as ex:

        return ImagePredictionResult(
            file_name = file_name,
            detector_type= detector_type,
            errors= str(ex),
            has_errors=True)
    

    finally:
        # Change the working directory back to the original directory
        os.chdir(original_dir)
