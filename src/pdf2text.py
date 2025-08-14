import pymupdf  # Imports the PyMuPDF library
import os  # Imports the os module for file and directory operations

# Define the directory containing the PDF files
source_directory = "/home/admin/Downloads"

# Loop through all files in the specified directory
for filename in os.listdir(source_directory):
    # Check if the file is a PDF
    if filename.endswith(".pdf"):
        # Construct the full path for the input PDF file
        input_pdf_path = os.path.join(source_directory, filename)

        # Create the corresponding output text file name
        # Replaces the '.pdf' extension with '.txt'
        output_txt_filename = os.path.splitext(filename)[0] + ".txt"

        # Construct the full path for the output text file
        output_txt_path = os.path.join(source_directory, output_txt_filename)

        try:
            # Open the PDF document
            doc = pymupdf.open(input_pdf_path)

            # Open the output text file in write mode with UTF-8 encoding
            with open(output_txt_path, "w", encoding="utf-8") as output_file:
                # Iterate through the pages of the document
                for page in doc:
                    text = page.get_text()  # Get plain text from the page
                    output_file.write(text)  # Write the text to the file
                    output_file.write("\n")  # Add a newline for page separation

            # Close the PDF document
            doc.close()

            print(f"Successfully converted '{filename}' to '{output_txt_filename}'")

        except Exception as e:
            print(f"Error processing '{filename}': {e}")

print("\nAll PDF files in the directory have been processed. ✨")
