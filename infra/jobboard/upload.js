
const FILE_UPLOAD_URL = "http://localhost:8080/addBinary";

async function addBinary(file) {
    console.log("addBinary ", file);

    const formData = new FormData();
    formData.append("binary", file);

    const response = await fetch(FILE_UPLOAD_URL, {
        method: "POST",
        body: formData
    });

    console.log('file uploaded response', response);
}

$(function() {
    console.log("document ready");

    // setup file upload form
    $("#upload_form").submit(function(event) {
        event.preventDefault();

        const form = event.target;
        const input = form[0];
        const files = input.files;

        if (files.length) {
            addBinary(files[0]);
        } else {
            console.error("Please select a file first");
        }
    });

});
