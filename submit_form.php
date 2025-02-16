<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Collect form data
    $name = htmlspecialchars($_POST['name']);
    $email = htmlspecialchars($_POST['email']);
    $message = htmlspecialchars($_POST['message']);

    // Set up email details
    $to = "xenoraai@gmail.com"; // Your email address
    $subject = "New Contact Form Submission from XenoraAI";
    $body = "Name: $name\nEmail: $email\nMessage:\n$message";
    $headers = "From: $email";

    // Send the email
    if (mail($to, $subject, $body, $headers)) {
        echo "<script>alert('Message sent successfully!'); window.location.href = 'contact.html';</script>";
    } else {
        echo "<script>alert('Failed to send message. Please try again.'); window.location.href = 'contact.html';</script>";
    }
} else {
    echo "<script>alert('Invalid request method.'); window.location.href = 'contact.html';</script>";
}
?>