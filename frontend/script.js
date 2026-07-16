async function predictWord(){

    const input=document.getElementById("inputText").value;

    if(input===""){

        alert("Enter a word");

        return;

    }

    const response=await fetch(

        "http://127.0.0.1:8000/predict",

        {

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify({

                text:input

            })

        }

    );

    const data=await response.json();

    document.getElementById("outputText").innerHTML=data.prediction;

}