using UnityEngine;

namespace AutonomousParking.Common.Extensions
{
    // StandardTypeExtensions 클래스는 수치 관련 확장 메서드를 제공하는 정적 클래스입니다.
    public static class StandardTypeExtensions
    {
        // Normalize 메서드는 주어진 float 값(value)을 0과 1 사이의 값으로 정규화합니다.
        public static float Normalize(this float value, float min, float max)
        {
            // 정규화된 범위의 최소값과 최대값을 정의합니다.
            const float minNormalizedValue = 0f, maxNormalizedValue = 1f;
            // 입력된 값의 범위를 계산합니다.
            float range = max - min;
            // range가 0에 가깝다면 minNormalizedValue를 사용하고, 그렇지 않다면 범위로 정규화합니다.
            float normalizedValue = Mathf.Approximately(range, default) ? minNormalizedValue : (value - min) / range;
            // 정규화된 값이 항상 0과 1 사이가 되도록 클램핑합니다.
            normalizedValue = Mathf.Clamp(normalizedValue, minNormalizedValue, maxNormalizedValue);
            return normalizedValue;
        }

        public static float NormalizeWithNegative(this float value, float min, float max)
        {
            // 정규화된 범위의 최소값과 최대값을 정의합니다.
            const float minNormalizedValue = -1f, maxNormalizedValue = 1f;
            // 입력된 값의 범위를 계산합니다.
            float range = max - min;
            // range가 0에 가깝다면 minNormalizedValue를 사용하고, 그렇지 않다면 범위로 정규화합니다.
            float normalizedValue = Mathf.Approximately(range, default) ? minNormalizedValue : (value - min) / range;
            // 정규화된 값이 항상 0과 1 사이가 되도록 클램핑합니다.
            normalizedValue = Mathf.Clamp(normalizedValue, minNormalizedValue, maxNormalizedValue);
            return normalizedValue;
        }

        // ChangeBounds 메서드는 float 값의 범위를 변경합니다.
        public static float ChangeBounds(this float value, float oldMin, float oldMax, float newMin, float newMax) =>
            // 먼저 값의 범위를 정규화한 후, 새로운 범위로 조정합니다.
            value.Normalize(oldMin, oldMax) * (newMax - newMin) + newMin;

        // ChangeBounds 메서드는 int 값을 받아 새로운 float 범위로 변환합니다.
        public static float ChangeBounds(this int value, float oldMin, float oldMax, float newMin, float newMax) =>
            // int 값을 float로 캐스팅한 후, float 범위 변경 메서드를 사용합니다.
            ((float)value).ChangeBounds(oldMin, oldMax, newMin, newMax);
    }
}
