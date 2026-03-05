import os
import glob

def remove_pdfs_from_subfolders(base_dir="Utils/Dataset/Tester/plots_by_feature_png"):
    """Remove all PDF files from subfolders in the specified directory"""
    
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist!")
        return
    
    print(f"Scanning for PDF files in subfolders of: {base_dir}")
    
    total_pdfs_found = 0
    total_pdfs_removed = 0
    folders_processed = 0
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print("No subfolders found in the directory.")
        return
    
    print(f"Found {len(subdirs)} subfolders to process:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    # Ask for confirmation
    response = input(f"\nDo you want to delete all PDF files from these {len(subdirs)} subfolders? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    # Process each subfolder
    for subdir in subdirs:
        subfolder_path = os.path.join(base_dir, subdir)
        folders_processed += 1
        
        print(f"\n[{folders_processed}/{len(subdirs)}] Processing folder: {subdir}")
        
        # Find all PDF files in this subfolder
        pdf_pattern = os.path.join(subfolder_path, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        folder_pdfs_found = len(pdf_files)
        folder_pdfs_removed = 0
        total_pdfs_found += folder_pdfs_found
        
        if folder_pdfs_found == 0:
            print(f"  No PDF files found in {subdir}")
            continue
        
        print(f"  Found {folder_pdfs_found} PDF files in {subdir}")
        
        # Remove each PDF file
        for pdf_file in pdf_files:
            try:
                os.remove(pdf_file)
                folder_pdfs_removed += 1
                total_pdfs_removed += 1
                print(f"    ✓ Removed: {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"    ✗ Error removing {os.path.basename(pdf_file)}: {e}")
        
        print(f"  Removed {folder_pdfs_removed}/{folder_pdfs_found} PDF files from {subdir}")
    
    # Summary
    print(f"\n--- REMOVAL SUMMARY ---")
    print(f"Folders processed: {folders_processed}")
    print(f"Total PDF files found: {total_pdfs_found}")
    print(f"Total PDF files removed: {total_pdfs_removed}")
    
    if total_pdfs_removed == total_pdfs_found:
        print("✅ All PDF files successfully removed!")
    elif total_pdfs_removed > 0:
        print(f"⚠️  {total_pdfs_found - total_pdfs_removed} PDF files could not be removed")
    else:
        print("❌ No PDF files were removed")

def list_pdfs_only(base_dir="Utils/Dataset/Tester/plots_by_feature_png"):
    """List all PDF files without removing them (for preview)"""
    
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist!")
        return
    
    print(f"Listing PDF files in subfolders of: {base_dir}")
    
    total_pdfs = 0
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        subfolder_path = os.path.join(base_dir, subdir)
        pdf_pattern = os.path.join(subfolder_path, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        if pdf_files:
            print(f"\n{subdir}/ ({len(pdf_files)} PDFs):")
            for pdf_file in pdf_files:
                print(f"  - {os.path.basename(pdf_file)}")
            total_pdfs += len(pdf_files)
        else:
            print(f"\n{subdir}/ (no PDFs)")
    
    print(f"\nTotal PDF files found: {total_pdfs}")

# Main execution
if __name__ == "__main__":
    print("PDF Removal Script for plots_by_feature_png")
    print("=" * 50)
    
    # First, show what PDFs exist
    print("1. Listing existing PDF files...")
    list_pdfs_only()
    
    print("\n" + "=" * 50)
    
    # Then ask if user wants to remove them
    print("2. PDF Removal Process")
    remove_pdfs_from_subfolders()
