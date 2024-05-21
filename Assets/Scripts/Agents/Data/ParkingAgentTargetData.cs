using UnityEngine;

namespace AutonomousParking.Agents.Data
{
    public class ParkingAgentTargetData : MonoBehaviour
    {
        [field: SerializeField] public float ParkingRadius { get; private set; } = 0.5f;
        [field: SerializeField] public float ParkingAngle { get; private set; } = 6f;
        [field: SerializeField] public float PerfectParkingRadius { get; private set; } = 0.15f;
        [field: SerializeField] public float PerfectParkingAngle { get; private set; } = 2f;

        public Transform Transform { get; private set; }

        private readonly Color parkingZoneColor = Color.red;
        private readonly Color perfectParkingZoneColor = Color.white;

        private void Awake() => Transform = transform;

        private void OnRenderObject()
        {
            DrawParkingZone();
        }

        private void DrawParkingZone()
        {
            // Set the color for the parking zone
            GL.PushMatrix();
            GL.MultMatrix(transform.localToWorldMatrix);

            // Draw the parking zone
            GL.Begin(GL.LINES);
            GL.Color(parkingZoneColor);

            for (int i = 0; i < 360; i++)
            {
                float angle = i * Mathf.Deg2Rad;
                float nextAngle = (i + 1) * Mathf.Deg2Rad;
                GL.Vertex3(Mathf.Cos(angle) * ParkingRadius, 0, Mathf.Sin(angle) * ParkingRadius);
                GL.Vertex3(Mathf.Cos(nextAngle) * ParkingRadius, 0, Mathf.Sin(nextAngle) * ParkingRadius);
            }

            GL.End();

            // Set the color for the perfect parking zone
            GL.Begin(GL.LINES);
            GL.Color(perfectParkingZoneColor);

            for (int i = 0; i < 360; i++)
            {
                float angle = i * Mathf.Deg2Rad;
                float nextAngle = (i + 1) * Mathf.Deg2Rad;
                GL.Vertex3(Mathf.Cos(angle) * PerfectParkingRadius, 0, Mathf.Sin(angle) * PerfectParkingRadius);
                GL.Vertex3(Mathf.Cos(nextAngle) * PerfectParkingRadius, 0, Mathf.Sin(nextAngle) * PerfectParkingRadius);
            }

            GL.End();
            GL.PopMatrix();
        }
    }
}
