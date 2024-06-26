﻿using AutonomousParking.Agents.Data;
using AutonomousParking.Car;
using UnityEngine;
using AutonomousParking.ParkingLot.ObjectPlacers;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentInitializer : MonoBehaviour
    {
        [field: SerializeField] private ParkingAgentTargetData targetData;

        [field: SerializeField] private ParkingLotEnteringCarPlacer agentPlacer;
        [field: SerializeField] private ParkingLotAgentTargetPlacer targetPlacer;
        [field: SerializeField] private ParkingLotParkedCarsPlacer parkedCarsPlacer;

        public void InitializeExternal(ParkingAgent agent)
        {
            agent.TargetData = targetData;

            agent.AgentPlacer = agentPlacer;
            agent.TargetPlacer = targetPlacer;
            agent.ParkedCarsPlacer = parkedCarsPlacer;
        }

        public void InitializeData(ParkingAgent agent)
        {
            agent.CarData = GetComponentInChildren<CarData>();
            agent.CollisionData = new ParkingAgentCollisionData();
            agent.RewardData.Initialize();

            agent.AgentData.Agent = agent;
            agent.AgentData.Rigidbody = GetComponent<Rigidbody>();
            agent.AgentData.Transform = GetComponent<Transform>();
        }

        public void InitializeComponents(ParkingAgent agent)
        {
            CarData carData = agent.CarData;
            ParkingAgentData agentData = agent.AgentData;
            ParkingAgentTargetTrackingData targetTrackingData = agent.TargetTrackingData;
            ParkingAgentCollisionData collisionData = agent.CollisionData;

            agent.ActionsHandler = new ParkingAgentActionsHandler(carData, agentData);
            agent.MetricsCalculator = new ParkingAgentMetricsCalculator(agentData, targetData, targetTrackingData);
            agent.RewardCalculator =
                new ParkingAgentRewardCalculator(collisionData, agentData, agent.RewardData, targetTrackingData);
            agent.ObservationsCollector = new ParkingAgentObservationsCollector(agentData, targetData, targetTrackingData);
            agent.CollisionsHandler = GetComponent<ParkingAgentCollisionsHandler>().Initialize(collisionData);
            agent.StatsRecorder = new ParkingAgentStatsRecorder(collisionData, targetTrackingData);
        }
    }
}