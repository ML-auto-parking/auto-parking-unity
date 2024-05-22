using Unity.MLAgents;
using UnityEngine;
using AutonomousParking.Agents;
using System;

namespace AutonomousParking.Agents.Data
{
    public class ParkingSpotArea : MonoBehaviour
    {
        [field: SerializeField] public ParkingAgent Agent { get; private set; }
        public void Start()
        { }
        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Agent"))
            {
                Debug.Log("Agent가 영역안에 들어왔습니다.");
                Agent.EnterArea(this.transform);
            }
        }
    }
}