// Syllabus JavaScript

document.addEventListener('DOMContentLoaded', () => {
    const uploadPdf = document.getElementById('uploadPdf');
    const pdfInput = document.getElementById('pdfInput');
    const form = document.getElementById('syllabusUploadForm');
    
    
    if (uploadPdf) {
        uploadPdf.addEventListener('click', () => {
            pdfInput.click();
        });
    }
    
    if (pdfInput) {
        pdfInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type === 'application/pdf') {
                // Real upload: submit the form to Flask route (/syllabus/upload)
                if (uploadPdf) {
                    uploadPdf.disabled = true;
                    uploadPdf.textContent = 'Uploading...';
                }
                form?.submit();
            }
        });
    }
});
