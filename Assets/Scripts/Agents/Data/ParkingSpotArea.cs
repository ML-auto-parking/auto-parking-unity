using Unity.MLAgents;
using UnityEngine;

namespace AutonomousParking.Agents.Data
{
    public class ParkingSpotArea : MonoBehaviour
    {
        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Agent"))
            {
                Debug.Log("Agent가 영역안에 들어왔습니다.");
            }
        }
    }
}