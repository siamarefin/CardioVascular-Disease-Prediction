"use client";

import React, { useState } from "react";
import axios from "axios";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { motion } from "framer-motion";

export default function HeartTest() {
  const [formData, setFormData] = useState({
    ap_hi: 120,
    ap_lo: 80,
    cholesterol: 1,
    age_years: 30,
    bmi: 25,
  });
  const [result, setResult] = useState(null);
   const [loading, setLoading] = useState(false);
   
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handlePredict = async () => {
    try {
      const res = await axios.post("http://localhost:8000/predict", formData);
      setLoading(true);
      setTimeout(() => {
        setLoading(false);
      }, 2000); // Simulate loading for 2 seconds
      setResult(res.data.prediction.predicted_cardio);
      // console.log("Prediction result:", res.data.prediction.predicted_cardio);

      
    } catch (err) {
      console.error(err);
    }
  };

  const getMessage = () => {
    if (result === 1) {
      return {
        text: "⚠️ You may have a high risk of heart disease. Please consult a doctor.",
        color: "text-red-600",
      };
    } else if (result === 0) {
      return {
        text: "✅ Your heart condition seems fine. Keep maintaining a healthy lifestyle!",
        color: "text-green-600",
      };
    }
    return null;
  };

  const feedback = getMessage();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-rose-100 to-red-200 p-4">
      <Card className="w-full max-w-md shadow-xl border-rose-300">
        <CardHeader>
          <CardTitle className="text-2xl text-center text-red-600 font-bold">
            ❤️ Heart Health Predictor
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {["ap_hi", "ap_lo", "cholesterol", "age_years", "bmi"].map((field) => (
            <div key={field}>
              <Label htmlFor={field} className="capitalize">
                {field.replace("_", " ")}
              </Label>
              <Input
                id={field}
                name={field}
                type="number"
                value={formData[field]}
                onChange={handleChange}
                className="bg-white"
              />
            </div>
          ))}
          <Button
            onClick={handlePredict}
            className="w-full bg-red-600 hover:bg-red-700 text-white"
          >
            Predict
          </Button>
          
{loading ? (
  <iframe className="m-auto" src="https://lottie.host/embed/f5caae7d-e69a-43e6-a3b7-c7e49f178f9a/NTrS6sX2pI.lottie"></iframe>
) : (
  result !== null && feedback && (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`mt-4 text-center text-lg font-semibold ${feedback.color}`}
    >
      {feedback.text}
    </motion.div>
  )
)}
        </CardContent>
      </Card>

  
  <h1 className="text-2xl font-bold text-red-600 mt-8">
    Developed by Robin and Siam
  </h1>
      <p className="text-gray-600 mt-2">
        This project is built using Next.js, React, and AI/ML techniques.
      </p>
    </div>
  );
}
