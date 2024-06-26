﻿using AutonomousParking.Agents.Data;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentObservationsCollector
    {
        private readonly ParkingAgentData agentData;
        private readonly ParkingAgentTargetData targetData;
        private readonly ParkingAgentTargetTrackingData targetTrackingData;

        public ParkingAgentObservationsCollector(ParkingAgentData agentData, ParkingAgentTargetData targetData, ParkingAgentTargetTrackingData targetTrackingData)
        {
            this.agentData = agentData;
            this.targetData = targetData;
            this.targetTrackingData = targetTrackingData;
        }

        public void CollectAgentTransformObservations(VectorSensor sensor)
        {
            Vector3 agentPosition = agentData.Transform.position;
            sensor.AddObservation(agentPosition.x);
            sensor.AddObservation(agentPosition.z);
            sensor.AddObservation(agentData.Transform.rotation.eulerAngles.y);
        }

        public void CollectAgentVelocityObservations(VectorSensor sensor)
        {
            sensor.AddObservation(agentData.Rigidbody.velocity.x);
            sensor.AddObservation(agentData.Rigidbody.velocity.z);
            sensor.AddObservation(agentData.Rigidbody.angularVelocity.y);
        }

        public void CollectTargetTransformObservations(VectorSensor sensor)
        {
            Transform targetTransform = targetData.Transform;
            Vector3 targetPosition = targetTransform.position;
            sensor.AddObservation(targetPosition.x);
            sensor.AddObservation(targetPosition.z);
            sensor.AddObservation(targetTransform.rotation.eulerAngles.y);
        }

        public void CollectAgentActionObservations(VectorSensor sensor)
        {
            sensor.AddObservation(agentData.CurrentWheelTorque);
            sensor.AddObservation(agentData.CurrentSteeringAngle);
        }

        public void CollectParkingSuccessObservations(VectorSensor sensor)
        {
            sensor.AddObservation(targetTrackingData.IsParked);
            sensor.AddObservation(targetTrackingData.IsPerfectlyParked);
        }
    }
}