using System;
using UnityEngine;

namespace AutonomousParking.Agents.Data
{
    [Serializable]
    public class ParkingAgentData
    {
        [field: SerializeField] public int MinStepToStartParking { get; private set; }
        public int MaxStepToStartParking => Agent.MaxStep;
        public int StepCount => Agent.StepCount;
        public bool HasReachedMaxStep => Agent.StepCount == Agent.MaxStep;
        public bool isInTargetArea = false;

        public ParkingAgent Agent { get; set; }
        public Rigidbody Rigidbody { get; set; }
        public Transform Transform { get; set; }
        public float CurrentWheelTorque { get; set; } = 0;
        public float CurrentSteeringAngle { get; set; } = 0;

        public void Reset() => Rigidbody.velocity = Rigidbody.angularVelocity = default;
    }
}