using System;
using AutonomousParking.Car;
using AutonomousParking.Car.UserInput;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentActionsHandler
    {
        private readonly CarData carData;
        private readonly CarUserInputInterpreter interpreter;

        // 이산 값의 범위 설정 (예: 10개의 이산 값으로 정밀도 증가)
        private const int NumDiscreteWheelTorqueValues = 5;
        private const int NumDiscreteSteeringAngleValues = 5;

        public ParkingAgentActionsHandler(CarData carData)
        {
            this.carData = carData;
            interpreter = new CarUserInputInterpreter(carData);
        }

        public void HandleInputActions(ActionBuffers actions)
        {
            // 이산 액션 배열 크기 확인
            if (actions.DiscreteActions.Length < 2)
            {
                Debug.LogError("Expected at least 3 discrete actions.");
                return;
            }

            // 이산 액션 값을 연속 값으로 변환하여 사용
            var discreteWheelTorque = actions.DiscreteActions[0];
            var discreteSteeringAngle = actions.DiscreteActions[1];

            carData.CurrentWheelTorque = interpreter.InterpretAsWheelTorque(ConvertDiscreteToContinuous(discreteWheelTorque, -1f, 1f, NumDiscreteWheelTorqueValues));
            carData.CurrentSteeringAngle = interpreter.InterpretAsSteeringAngle(ConvertDiscreteToContinuous(discreteSteeringAngle, -1f, 1f, NumDiscreteSteeringAngleValues));
        }

        public void HandleHeuristicInputDiscreteActions(in ActionSegment<int> discreteActionsOut)
        {
            if (discreteActionsOut.Length < 2)
            {
                Debug.LogError("Expected at least 3 discrete actions out.");
                return;
            }


            discreteActionsOut[0] = ConvertContinuousToDiscrete(CarUserInputData.WheelTorque, -1f, 1f, NumDiscreteWheelTorqueValues);
            discreteActionsOut[1] = ConvertContinuousToDiscrete(CarUserInputData.SteeringAngle, -1f, 1f, NumDiscreteSteeringAngleValues);
        }

        // 연속적인 값을 이산적인 값으로 변환하는 메서드
        private int ConvertContinuousToDiscrete(float value, float minValue, float maxValue, int numDiscreteValues)
        {
            if (numDiscreteValues <= 1) return 0;

            // Calculate the step size based on the given minValue, maxValue, and numDiscreteValues
            float low = minValue + (maxValue - minValue) / (2 * (numDiscreteValues - 1));
            float stepSize = (maxValue - minValue) / (numDiscreteValues - 1);
            // Iterate through the number of discrete values to find the correct range
            for (int i = 0; i < numDiscreteValues; i++)
            {
                float threshold = low + stepSize * i;
                if (value <= threshold)
                {
                    return i;
                }
            }

            // If value exceeds the maxValue, return the highest discrete value
            return numDiscreteValues - 1;
        }

        // 이산적인 값을 연속적인 값으로 변환하는 메서드
        private float ConvertDiscreteToContinuous(int discreteValue, float minValue, float maxValue, int numDiscreteValues)
        {
            if (numDiscreteValues <= 1) return minValue;

            // Calculate the step size based on the given minValue, maxValue, and numDiscreteValues
            float stepSize = (maxValue - minValue) / (numDiscreteValues - 1);

            // Calculate the continuous value for the given discrete value
            float continuousValue = minValue + stepSize * discreteValue;

            return continuousValue;
        }
    }
}
