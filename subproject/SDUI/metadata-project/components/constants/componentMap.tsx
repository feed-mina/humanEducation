//  @@@@ 2026-02-07 추가 컴포넌트들을 한곳에 관리하는 파일
import InputField from "@/components/fields/InputField";
import TextField from "@/components/fields/TextField";
import ButtonField from "@/components/fields/ButtonField";
import ImageField from "@/components/fields/ImageField";
import SelectField from "@/components/fields/SelectField";
import PasswordField from "@/components/fields/PasswordField";
import TextAreaField from "@/components/fields/TextAreaField";
import EmailSelectField from "@/components/fields/EmailSelectField";
import EmotionSelectField from "@/components/fields/EmotionSelectField";
import RecordTimeComponent from "@/components/fields/RecordTimeComponent";
import DateTimePicker from "@/components/fields/DateTimePicker";
import TimeSelect from "@/components/fields/TimeSelect";
import TimeSlotRecord from "@/components/fields/TimeSlotRecord";
import AddressSearchGroup from "@/components/fields/AddressSearchGroup";
import {withRenderTrack} from "@/components/utils/withRenderTrack";
import Modal from "@/components/fields/Modal";
import AdminUserTable from "@/components/fields/AdminUserTable";
import AIChatComponent from "@/components/fields/AIChatComponent";
import AIChatComponentV2 from "@/components/fields/AIChatComponentV2";
import AIInterviewComponent from "@/components/fields/AIInterviewComponent";
import CheckboxField from "@/components/fields/CheckboxField";

const GroupComponent: React.FC<any> = ({ children }) => <>{children}</>;

export const componentMap: Record<string, React.FC<any>> = {
    MODAL: withRenderTrack(Modal,"Modal"),
    INPUT: withRenderTrack(InputField,"InputField"),
    TEXT: withRenderTrack(TextField,"TextField") ,
    PASSWORD: withRenderTrack(PasswordField,"PasswordField"),
    BUTTON: withRenderTrack(ButtonField,"ButtonField"),
    SNS_BUTTON: withRenderTrack(ButtonField,"ButtonField"),
    LINK_BUTTON: withRenderTrack(ButtonField,"ButtonField"),
    IMAGE: withRenderTrack(ImageField,"ImageField"),
    EMAIL_SELECT: withRenderTrack(EmailSelectField,"EmailSelectField"),
    EMOTION_SELECT: withRenderTrack(EmotionSelectField,"EmotionSelectField"),
    SELECT: withRenderTrack(SelectField,"SelectField"),
    TEXTAREA: withRenderTrack(TextAreaField,"TextAreaField"),
    TIME_RECORD_WIDGET: withRenderTrack(RecordTimeComponent,"RecordTimeComponent"),
    DATETIME_PICKER: withRenderTrack(DateTimePicker,"DateTimePicker"),
    TIME_SELECT: withRenderTrack(TimeSelect,"TimeSelect"),
    TIME_SLOT_RECORD: withRenderTrack(TimeSlotRecord,"TimeSlotRecord"),
    ADDRESS_SEARCH_GROUP : withRenderTrack(AddressSearchGroup, "AddressSearchGroup"),
    GROUP:withRenderTrack(GroupComponent, "GroupField"),
    ADMIN_USER_TABLE: withRenderTrack(AdminUserTable, "AdminUserTable"),
    AI_CHAT: withRenderTrack(AIChatComponent, "AIChatComponent"),
    AI_CHAT_V2: withRenderTrack(AIChatComponentV2, "AIChatComponentV2"),
    AI_INTERVIEW: withRenderTrack(AIInterviewComponent, "AIInterviewComponent"),
    CHECKBOX: withRenderTrack(CheckboxField, "CheckboxField"),
};