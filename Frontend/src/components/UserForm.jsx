import { useState } from "react";
import { User, Calendar } from "lucide-react";

export default function UserForm({ onSubmit }) {
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [errors, setErrors] = useState({});

  const handleSubmit = (e) => {
    e.preventDefault();

    const newErrors = {};

    if (!name.trim()) {
      newErrors.name = "Name is required";
    }

    const ageNum = parseInt(age, 10);
    if (!age || isNaN(ageNum) || ageNum < 1 || ageNum > 120) {
      newErrors.age = "Please enter a valid age (1-120)";
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setErrors({});
    onSubmit({ name: name.trim(), age: ageNum });
  };

  return (
    <div className="w-100">
      <div className="text-center mb-4">
        <h5 className="d-flex justify-content-center align-items-center gap-2">
          <User size={20} />
          User Information
        </h5>
        <p className="text-muted mb-0">
          Please provide your details to get started
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Full Name */}
        <div className="mb-3">
          <label htmlFor="name" className="form-label">
            Full Name
          </label>
          <input
            id="name"
            type="text"
            className={`form-control ${errors.name ? "is-invalid" : ""}`}
            placeholder="Enter your full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          {errors.name && (
            <div className="invalid-feedback">{errors.name}</div>
          )}
        </div>

        {/* Age */}
        <div className="mb-3">
          <label
            htmlFor="age"
            className="form-label d-flex align-items-center gap-2"
          >
            <Calendar size={16} /> Age
          </label>
          <input
            id="age"
            type="number"
            className={`form-control ${errors.age ? "is-invalid" : ""}`}
            placeholder="Enter your age"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            min="1"
            max="120"
          />
          {errors.age && (
            <div className="invalid-feedback">{errors.age}</div>
          )}
        </div>

        {/* Submit */}
        <button type="submit" className="btn btn-primary w-100">
          Continue to Video Recording
        </button>
      </form>
    </div>
  );
}