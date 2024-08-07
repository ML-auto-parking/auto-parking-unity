﻿using System;
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
            
            carData.CurrentWheelTorque = interpreter.InterpretAsWheelTorque(agentData.CurrentWheelTorque);
            carData.CurrentSteeringAngle = interpreter.InterpretAsSteeringAngle(agentData.CurrentSteeringAngle);
        }

        public void HandleHeuristicInputContinuousActions(in ActionSegment<float> continuousActionsOut)
        {
            continuousActionsOut[0] = CarUserInputData.WheelTorque;
            continuousActionsOut[1] = CarUserInputData.SteeringAngle;
        }
    }
}
