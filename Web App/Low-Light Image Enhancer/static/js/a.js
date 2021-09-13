let base64Image;
function readURL(input) {
  if (input.files && input.files[0]) {

    let reader = new FileReader();
            reader.onload = function(e) {
                let dataURL = reader.result;
                input.attr("src", dataURL);
                base64Image = dataURL.replace("data:image/png;base64,","");
                console.log(base64Image);
                 $('.image-upload-wrap').hide();
                 $('.file-upload-image').attr('src', e.target.result);
                 $('.file-upload-content').show();
                 $('.image-title').html(input.files[0].name);
            }
            reader.readAsDataURL(input.files[0]);


  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
}
$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});
