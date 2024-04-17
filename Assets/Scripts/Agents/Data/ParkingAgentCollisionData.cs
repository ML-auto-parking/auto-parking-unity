using AutonomousParking.Common.Enumerations;

namespace AutonomousParking.Agents.Data
{
    // ParkingAgentCollisionData 클래스는 주차 에이전트의 충돌 관련 데이터를 저장하고 관리합니다.
    public class ParkingAgentCollisionData
    {
        // CollisionTag는 충돌이 발생했을 때 해당 충돌의 유형을 저장하는 속성입니다.
        // 예를 들어, 벽이나 다른 차량과의 충돌 등 다양한 충돌 유형을 Tag 열거형으로 표현할 수 있습니다.
        public Tag CollisionTag { get; set; }

        // IsAnyCollision 속성은 충돌이 발생했는지 여부를 나타냅니다.
        // CollisionTag가 기본값(default)이 아니면 true를 반환하며, 이는 어떤 종류의 충돌이 발생했음을 의미합니다.
        public bool IsAnyCollision => CollisionTag != default;
        
        // Reset 메서드는 충돌 데이터를 초기 상태로 재설정합니다.
        // 이를 통해 CollisionTag를 기본값으로 설정하여 충돌 상태를 초기화합니다.
        public void Reset() => CollisionTag = default;
    }
}
