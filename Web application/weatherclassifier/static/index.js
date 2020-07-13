$(document).ready(function(){
  $('form input').change(function () {
    $('form h1').text(this.files.length + " file selected");
  });
});