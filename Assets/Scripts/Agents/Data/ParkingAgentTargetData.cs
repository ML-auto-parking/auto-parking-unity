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
        private readonly Color perfectParkingZoneColor = Color.green;

        private Mesh parkingZoneMesh;
        private Mesh perfectParkingZoneMesh;

        private void Awake()
        {
            Transform = transform;
            parkingZoneMesh = CreateCircleMesh(ParkingRadius);
            perfectParkingZoneMesh = CreateCircleMesh(PerfectParkingRadius);
        }

        private void OnRenderObject()
        {
            DrawCircleMesh(parkingZoneMesh, parkingZoneColor);
            DrawCircleMesh(perfectParkingZoneMesh, perfectParkingZoneColor);
        }

        private Mesh CreateCircleMesh(float radius)
        {
            int segments = 360;
            Vector3[] vertices = new Vector3[segments + 1];
            int[] indices = new int[segments * 2];

            for (int i = 0; i <= segments; i++)
            {
                float angle = i * Mathf.Deg2Rad;
                vertices[i] = new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
            }

            for (int i = 0; i < segments; i++)
            {
                indices[i * 2] = i;
                indices[i * 2 + 1] = i + 1;
            }

            Mesh mesh = new Mesh();
            mesh.vertices = vertices;
            mesh.SetIndices(indices, MeshTopology.Lines, 0);
            return mesh;
        }

        private void DrawCircleMesh(Mesh mesh, Color color)
        {
            Material material = new Material(Shader.Find("Hidden/Internal-Colored"));
            material.SetPass(0);
            material.color = color;
            Graphics.DrawMeshNow(mesh, Transform.localToWorldMatrix);
        }
    }
}
