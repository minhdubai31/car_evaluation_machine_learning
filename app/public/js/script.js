$(function () {
    $('#form1').submit(function (e) { 
        e.preventDefault();
        resultDiv = $('#result').hide();
        $('#submitBtn').prop('disabled', true).text('Evaluating...');

        method = $(this).prop('method');
        url = $(this).prop('action');

        $.ajax({
            type: method,
            url: url,
            data: $(this).serialize(),
            processData: false,
            success: function (response) {
                $('#submitBtn').text('Evaluated');
                $('#resetBtn').show();
                resultDiv.children('span').text('This car is ' + response.toUpperCase());
                resultDiv.slideDown();
                console.log(JSON.stringify(response));
            },
            error: function (error) {
                $('#submitBtn').text('Can\'t connect to server');
                $('#resetBtn').show();
            }
        });
    });

    $('#resetBtn').click((e) => {
        $('#resetBtn').hide();
        $('#submitBtn').prop('disabled', false).text('Evaluate');
        $('#result').slideUp();
    });

    $('select').each((index, elem) =>{
        $(elem).change((e)=> {
            $('#resetBtn').hide();
            $('#submitBtn').prop('disabled', false).text('Evaluate');
            $('#result').slideUp();
        });
    });

    $('#csv_file').change((e) => {
        $('#form2').submit();
    })
});
