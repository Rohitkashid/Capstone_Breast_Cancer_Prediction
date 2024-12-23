const firebaseConfig = {
    apiKey: "AIzaSyD2GDgIIRsNGwHCSNyX2Pr0a2OpXZY5VU4",
    authDomain: "contactform-9405d.firebaseapp.com",
    databaseURL: "https://contactform-9405d-default-rtdb.firebaseio.com",
    projectId: "contactform-9405d",
    storageBucket: "contactform-9405d.appspot.com",
    messagingSenderId: "1079776000802",
    appId: "1:1079776000802:web:5adb32f143a80671e2c37b"
  };
  firebase.initializeApp(firebaseConfig);
  
  var contactFormDB = firebase.database().ref("contactForm");

  document.getElementById("contactForm").addEventListener("submit",submitForm);

  function submitForm(e){
    e.preventDefault();
    var name = getElementVal('name');
    var email = getElementVal('email');
    var message =getElementVal('message');

    saveMessages(name, email, message);

  document.querySelector('.alert').style.display = 'block';

  setTimeout( () => {
    document.querySelector('.alert').style.display = 'none';
  }, 3000);

  document.getElementById("contactForm").reset();

  }

const saveMessages = (name, email, message) =>{
var newContactForm = contactFormDB.push();
newContactForm.set({
    name : name,
    email : email,
    message : message,
});
};



  const getElementVal = (id) => {
    return document.getElementById(id).value;
  };