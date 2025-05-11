import React from 'react';
import { useRouter } from 'next/router';

const Result: React.FC = () => {
  const router = useRouter();
  // Extract the prediction result from the query parameters
  const prediction = router.query.prediction as string;

  return (
    <div>
      <header>
        <h1>Diabetes Prediction Result</h1>
      </header>

      <div>
        <p>{prediction}</p>
      </div>

      <footer>Created by Sanka Vaas</footer>
    </div>
  );
};

export default Result;
