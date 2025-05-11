// Import necessary modules and components
import React, { useState, FormEvent } from 'react';
import { useRouter } from 'next/router';

const Index: React.FC = () => {
  const [formData, setFormData] = useState({
    pregnancies: 0,
    glucose: 0,
    blood_pressure: 0,
    skin_thickness: 0,
    insulin: 0,
    bmi: 0,
    dpf: 0,
    age: 0,
  });

  const router = useRouter();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: parseFloat(value),
    }));
  };

  const handlePrediction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        // Redirect to the result page
        router.push('/result');
      } else {
        console.error('Error predicting:', response);
      }
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <div>
      <header>
        <h1>Diabetes Prediction</h1>
      </header>

      <form method="POST" onSubmit={handlePrediction}>
        <label htmlFor="pregnancies">Number of Pregnancies:</label>
        <input
          type="number"
          name="pregnancies"
          className="flex-parent jc-center"
          placeholder="Number of pregnancies"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="glucose">Glucose:</label>
        <input
          type="number"
          name="glucose"
          className="flex-parent jc-center"
          placeholder="Your glucose level"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="blood_pressure">Blood Pressure:</label>
        <input
          type="number"
          name="blood_pressure"
          className="flex-parent jc-center"
          placeholder="Blood pressure"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="skin_thickness">Skin thickness:</label>
        <input
          type="number"
          name="skin_thickness"
          className="flex-parent jc-center"
          placeholder="Skin thickness"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="insulin">Insulin:</label>
        <input
          type="number"
          name="insulin"
          className="flex-parent jc-center"
          placeholder="Insulin level"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="bmi">BMI value:</label>
        <input
          type="number"
          name="bmi"
          className="flex-parent jc-center"
          placeholder="Body Mass Index (BMI) value"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="dpf">Diabetic Pedigree Function:</label>
        <input
          type="number"
          name="dpf"
          className="flex-parent jc-center"
          placeholder="Diabetic Pedigree Function"
          required
          onChange={handleInputChange}
        />

        <label htmlFor="age">Age:</label>
        <input
          type="number"
          name="age"
          className="flex-parent jc-center"
          placeholder="Your age"
          required
          onChange={handleInputChange}
        />

        <div className="flex-parent jc-center">
          <input type="submit" value="Predict" />
        </div>
      </form>

      <footer>Created by Sanka Vaas</footer>
    </div>
  );
};

export default Index;
