import React from 'react';
import "../../app/styles/field.css";

interface ModalProps {
    meta: any;
    onConfirm: () => void;
    onClose: () => void;
}

const Modal = ({ meta, onConfirm, onClose }: ModalProps) => {
    const { label_text, props } = meta;
    console.log('Modal_props',props);
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-box" onClick={(e) => e.stopPropagation()}>
                <h2 className="modal-title">{label_text}</h2>
                {/*<p className="modal-content">{props.content}</p>*/}
                <button className="modal-btn" onClick={onConfirm}>
                    {/*{props.button_text || "확인"}*/}
                </button>
            </div>
        </div>
    );
};

export default Modal;