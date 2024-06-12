using System;
using AutonomousParking.Car;
using AutonomousParking.Car.UserInput;
using Unity.MLAgents.Actuators;
using UnityEngine;
using AutonomousParking.Agents.Data;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentActionsHandler
    {
        private readonly CarData carData;
        private readonly CarUserInputInterpreter interpreter;
        private readonly ParkingAgentData agentData;

        public ParkingAgentActionsHandler(CarData carData, ParkingAgentData agentData)
        {
            this.carData = carData;
            interpreter = new CarUserInputInterpreter(carData);
            this.agentData = agentData;
        }

        public void HandleInputActions(ActionBuffers actions)
        {
            agentData.CurrentWheelTorque = actions.ContinuousActions[0];
            agentData.CurrentSteeringAngle = actions.ContinuousActions[1];
            
            Debug.Log($"AgentData: WheelTorque={agentData.CurrentWheelTorque}, SteeringAngle={agentData.CurrentSteeringAngle}");
            Debug.Log($"CarData: WheelTorque={carData.CurrentWheelTorque}, SteeringAngle={carData.CurrentSteeringAngle}");

            carData.CurrentWheelTorque = interpreter.InterpretAsWheelTorque(agentData.CurrentWheelTorque);
            carData.CurrentSteeringAngle = interpreter.InterpretAsSteeringAngle(agentData.CurrentSteeringAngle);

            Debug.Log($"Interpreted Wheel Torque: {carData.CurrentWheelTorque}");
            Debug.Log($"Interpreted Steering Angle: {carData.CurrentSteeringAngle}");

            Debug.Log($"CarData: WheelTorque={carData.CurrentWheelTorque}, SteeringAngle={carData.CurrentSteeringAngle}");
        }

        public void HandleHeuristicInputContinuousActions(in ActionSegment<float> continuousActionsOut)
        {
            continuousActionsOut[0] = CarUserInputData.WheelTorque;
            continuousActionsOut[1] = CarUserInputData.SteeringAngle;
        }
    }
}
