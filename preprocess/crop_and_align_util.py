import os
import shutil

def move_matching_files():
    # Define the directories
    part2_dir = '/home/stu13/s8/ia3494/exp_ai_assignments/data/part2'
    cropped_part2_dir = '/home/stu13/s8/ia3494/exp_ai_assignments/data/part2_cropped'
    completed_part2_dir = '/home/stu13/s8/ia3494/exp_ai_assignments/data/part2_completed'

    # Ensure the completed_part2 directory exists
    if not os.path.exists(completed_part2_dir):
        os.makedirs(completed_part2_dir)

    # Get lists of files in part2 and cropped_part2
    part2_files = set(os.listdir(part2_dir))
    cropped_part2_files = set(os.listdir(cropped_part2_dir))

    # Find the matching files (based on name)
    matching_files = part2_files.intersection(cropped_part2_files)

    # Move the matching files from part2 to completed_part2
    for file_name in matching_files:
        # Construct full paths
        part2_file_path = os.path.join(part2_dir, file_name)
        completed_part2_file_path = os.path.join(completed_part2_dir, file_name)

        # Move the file
        shutil.move(part2_file_path, completed_part2_file_path)
        print(f"Moved: {file_name}")

    print(f"Completed moving {len(matching_files)} files to {completed_part2_dir}.")

if __name__ == "__main__":
    move_matching_files()
