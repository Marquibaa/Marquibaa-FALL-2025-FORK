#include <Servo.h>

Servo myServo;

void setup() 
{
  Serial.begin(9600);
  myServo.attach(9);       // keep attached so it can hold when powered
}

void loop() 
{
  if (Serial.available()) 
  {
    int angle = Serial.parseInt();
    // parseInt returns 0 if no number found
    if (angle >= 0 && angle <= 180) 
    {
      myServo.write(angle);
      Serial.print("OK:");
      Serial.println(angle);
    }
    // clear the rest of the line
    Serial.read(); // consume newline or leftover
  }
}
